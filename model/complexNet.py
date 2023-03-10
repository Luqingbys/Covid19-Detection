import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window


def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    ''' 
    使用STFT核初始化卷积/逆卷积模块（即ConvSTFT/ConviSTFT）
    返回: 经STFT得到的kernel和window
    '''
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    else:
        # **0.5   这里window是一个长度为win_len的一维张量
        window = get_window(win_type, win_len, fftbins=True)

    N = fft_len
    # np.fft.rfft(a) 计算实际输入的一维离散傅里叶变换
    # fourier_basis尺寸: (400, 257), dtype='complex'
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)  # 实部, dtype='float', (400, 257)
    imag_kernel = np.imag(fourier_basis)  # 虚部, dtype='float', (400, 257)
    # 将实部和虚部按列拼接，扩展列，经转置，尺寸为(514, 400)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T

    # invers默认为False
    if invers:
        # 求kernel矩阵（二维张量）的伪逆，尺寸不变
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel*window  # kernel: (514, 400), window: (400,)
    kernel = kernel[:, None, :]  # 同样地，kernel维度扩展成三维，返回的时候转为tensor类型
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None, :, None].astype(np.float32))


class ConvSTFT(nn.Module):
    '''
    短时傅立叶变换卷积模块
    win_type='hamming'
    '''

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real', fix=True):
        ''' 
        win_len:
        win_inc:
        fft_len: 
        win_type: 
        '''
        super(ConvSTFT, self).__init__()

        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len

        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        # self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)  # 将kernel注册为参数'weight', 期望将其保存
        self.feature_type = feature_type  # 类型，默认为real，也可以是complex
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:  # inputs: (1, 1, 128000)
            inputs = torch.unsqueeze(inputs, 1)
        print('in ConvSTFT: ', inputs.shape)
        inputs = F.pad(
            inputs, [self.win_len-self.stride, self.win_len-self.stride])  # F.pad()填充，从(1, 1, 128000)的第二维度左边、右边填充 self.win_len-self.stride=160 个0
        # 卷积核kernel使用上面init_kernels()得到的核，相比于conv2d，conv1d接收的张量尺寸中有一维被忽略
        outputs = F.conv1d(inputs, self.weight, stride=self.stride)

        if self.feature_type == 'complex':
            return outputs
        else:
            dim = self.dim//2+1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            # 从笛卡尔坐标映射到极坐标，real、imag变换为幅度mags、相位phase
            mags = torch.sqrt(real**2+imag**2)
            phase = torch.atan2(imag, real)
            return mags, phase


class NavieComplexLSTM(nn.Module):
    '''复LSTM'''
    def __init__(self, input_size, hidden_size, projection_dim=None, bidirectional=False, batch_first=False):
        super(NavieComplexLSTM, self).__init__()

        self.input_dim = input_size//2
        self.rnn_units = hidden_size//2
        # real_lstm、imag_lstm本质上都是朴素的LSTM，只不过前者接受实部，后者接收虚部
        self.real_lstm = nn.LSTM(self.input_dim, self.rnn_units,
                                 num_layers=1, bidirectional=bidirectional, batch_first=False)
        self.imag_lstm = nn.LSTM(self.input_dim, self.rnn_units,
                                 num_layers=1, bidirectional=bidirectional, batch_first=False)
        if bidirectional:
            bidirectional = 2
        else:
            bidirectional = 1
        if projection_dim is not None:
            self.projection_dim = projection_dim//2
            self.r_trans = nn.Linear(
                self.rnn_units*bidirectional, self.projection_dim)
            self.i_trans = nn.Linear(
                self.rnn_units*bidirectional, self.projection_dim)
        else:
            self.projection_dim = None

    def forward(self, inputs):
        if isinstance(inputs, list):
            real, imag = inputs
        elif isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs, -1)
        r2r_out = self.real_lstm(real)[0]
        r2i_out = self.imag_lstm(real)[0]
        i2r_out = self.real_lstm(imag)[0]
        i2i_out = self.imag_lstm(imag)[0]
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out
        if self.projection_dim is not None:
            real_out = self.r_trans(real_out)
            imag_out = self.i_trans(imag_out)
        # print(real_out.shape,imag_out.shape)
        return [real_out, imag_out]

    def flatten_parameters(self):
        self.imag_lstm.flatten_parameters()
        self.real_lstm.flatten_parameters()


class cPReLU(nn.Module):
    '''复激活层'''

    def __init__(self, complex_axis=1):
        super(cPReLU, self).__init__()
        # nn.PReLU()，一种激活函数，类似于ReLU()，表达式为Parametric ReLU，PReLU(x) = max(x, 0) + a*max(0, x)，其中参数a可学习
        # https://blog.csdn.net/flyfish1986/article/details/106649011
        self.r_prelu = nn.PReLU()
        self.i_prelu = nn.PReLU()
        self.complex_axis = complex_axis

    def forward(self, inputs):
        # torch.chunk(tensor, num, axis)，在tensor的第axis维度上进行分块，inputs是一个复张量，complex_axis指的是虚部维度，得到复张量的实部和虚部
        real, imag = torch.chunk(inputs, 2, self.complex_axis)
        real = self.r_prelu(real)
        imag = self.i_prelu(imag)
        # torch.cat()，在指定维度上对tensor进行拼接，恢复到原来的inputs
        return torch.cat([real, imag], self.complex_axis)


class ComplexConv2d(nn.Module):
    '''复卷积层'''
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=1,
        groups=1,
        causal=True,
        complex_axis=1,
    ):
        '''
            in_channels: real+imag，输入通道数
            out_channels: real+imag ，输出通道数
            kernel_size : input [B,C,D,T] kernel size in [D,T]，卷积核尺寸
            stride: 步长
            padding: input [B,C,D,T] padding in [D,T]，填充
            dilation: 是否采用空洞卷积，1表示不采用
            groups: 决定是否采用分组卷积
            causal: if causal, will padding time dimension's left side,
                    otherwise both
            complex_axis: 默认为1，指的是在哪一个维度上将张量拆分为实部和虚部

        '''
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels//2 # 为输入张量通道数的一半，输入张量一半为实部一半为虚部
        self.out_channels = out_channels//2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal = causal
        self.groups = groups
        self.dilation = dilation
        self.complex_axis = complex_axis
        # 复卷积层本质上是朴素的卷积层
        self.real_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride, padding=[
                                   self.padding[0], 0], dilation=self.dilation, groups=self.groups)
        self.imag_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, self.stride, padding=[
                                   self.padding[0], 0], dilation=self.dilation, groups=self.groups)
        
        # torch.nn.init.normal_(tensor, mean, std)，用正态分布给张量初始化
        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)
        # torch.nn.init.constant_(tensor, val)，使用val来填充tensor
        nn.init.constant_(self.real_conv.bias, 0.)
        nn.init.constant_(self.imag_conv.bias, 0.)


    def forward(self, inputs):
        if self.padding[1] != 0 and self.causal:
            inputs = F.pad(inputs, [self.padding[1], 0, 0, 0]) # if causal，inputs左边补零，个数为self.padding[1]
        else:
            inputs = F.pad(inputs, [self.padding[1], self.padding[1], 0, 0]) # 不然inputs两边都补零，个数均为self.padding[1]

        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)

        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis) # chunk()数组拆分，沿着指定维度拆分为指定数量的tensor
            print('before this ComplexConv2d: ')
            print('实部：', real.shape)
            print('虚部：', imag.shape)
            real2real = self.real_conv(real,)
            imag2imag = self.imag_conv(imag,)

            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis) # 沿着复数轴将实部和虚部拼接起来，real和imag均为四维张量

        return out


class ComplexBatchNorm(torch.nn.Module):
    '''复批标准化层'''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, complex_axis=1):
        super(ComplexBatchNorm, self).__init__()
        self.num_features = num_features//2
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.complex_axis = complex_axis

        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Br = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Bi = torch.nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            self.register_parameter('Wii', None)
            self.register_parameter('Br',  None)
            self.register_parameter('Bi',  None)

        if self.track_running_stats:
            self.register_buffer('RMr',  torch.zeros(self.num_features))
            self.register_buffer('RMi',  torch.zeros(self.num_features))
            self.register_buffer('RVrr', torch.ones(self.num_features))
            self.register_buffer('RVri', torch.zeros(self.num_features))
            self.register_buffer('RVii', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr',                 None)
            self.register_parameter('RMi',                 None)
            self.register_parameter('RVrr',                None)
            self.register_parameter('RVri',                None)
            self.register_parameter('RVii',                None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9)  # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert(xr.shape == xi.shape)
        assert(xr.size(1) == self.num_features)


    def forward(self, inputs):
        #self._check_input_dim(xr, xi)

        xr, xi = torch.chunk(inputs, 2, axis=self.complex_axis) # 切分出来实部和虚部
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i != 1]
        vdim = [1] * xr.dim()
        vdim[1] = xr.size(1)

        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr-Mr, xi-Mi

        #
        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        #
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr = Vrr + self.eps
        Vri = Vri
        Vii = Vii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, -1, Vri, Vri)
        s = delta.sqrt()
        t = (tau + 2*s).sqrt()

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst = (s * t).reciprocal()
        Urr = (s + Vii) * rst
        Uii = (s + Vrr) * rst
        Uri = (- Vri) * rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Wrr, Wri, Wii = self.Wrr.view(
                vdim), self.Wri.view(vdim), self.Wii.view(vdim)
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        outputs = torch.cat([yr, yi], self.complex_axis)
        return outputs


    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
            'track_running_stats={track_running_stats}'.format(**self.__dict__)


class ComplexNet(nn.Module):

    def __init__(
        self,
        rnn_layers=2,
        rnn_units=128,
        win_len=400,
        win_inc=100,
        fft_len=512,
        win_type='hann',
        masking_mode='E',
        use_clstm=True,
        use_cbn=True,
        kernel_size=5,
        kernel_num=[16, 32, 64, 128, 256, 256],
        ffn_input=(2, 10)
    ):
    # 原写法是win_type='hanning'，会报错，理由应该是scipy.signal.get_window(window, ...)中的window类型不包括hanning，类型支持hann、hamming等
    # https://vimsky.com/examples/usage/python-scipy.signal.get_window.html
        ''' 

            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag

        '''

        super(ComplexNet, self).__init__()

        # for fft
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        input_dim = win_len
        output_dim = win_len

        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size
        #self.kernel_num = [2, 8, 16, 32, 128, 128, 128]
        #self.kernel_num = [2, 16, 32, 64, 128, 256, 256]
        self.kernel_num = [2]+kernel_num # 默认是七个kernel_num
        self.masking_mode = masking_mode
        self.use_clstm = use_clstm
        self.ffn_input = ffn_input

        # bidirectional=True
        bidirectional = False
        fac = 2 if bidirectional else 1

        fix = True
        self.fix = fix
        # ConvSTFT
        self.stft = ConvSTFT(self.win_len, self.win_inc,
                             fft_len, self.win_type, 'complex', fix=fix)
        # # 逆ConvSTFT
        # self.istft = ConviSTFT(self.win_len, self.win_inc,
        #                        fft_len, self.win_type, 'complex', fix=fix)

        # 编码器，多次卷积块（每一个卷积块由多个卷积层+批标准化+激活函数构成）
        self.encoder = nn.ModuleList()
        # 解码器，多次逆卷积
        self.decoder = nn.ModuleList()
        # 添加编码器Encoder，默认添加6个，1个Encoder由1个复卷积层、复标准化层和复激活层组成
        for idx in range(len(self.kernel_num)-1):
            self.encoder.append(
                nn.Sequential(
                    #nn.ConstantPad2d([0, 0, 0, 0], 0),
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx+1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1)
                    ),
                    nn.BatchNorm2d(
                        self.kernel_num[idx+1]) if not use_cbn else ComplexBatchNorm(self.kernel_num[idx+1]),
                    nn.PReLU()
                )
            )
        hidden_dim = self.fft_len//(2**(len(self.kernel_num)))

        if self.use_clstm:
            rnns = []
            # rnns默认添加2个复LSTM（rnn_layers=2）
            for idx in range(rnn_layers):
                rnns.append(
                    NavieComplexLSTM(
                        input_size=hidden_dim *
                        self.kernel_num[-1] if idx == 0 else self.rnn_units,
                        hidden_size=self.rnn_units,
                        bidirectional=bidirectional,
                        batch_first=False,
                        projection_dim=hidden_dim *
                        self.kernel_num[-1] if idx == rnn_layers-1 else None,
                    )
                )
                # 复LSTM层
            self.enhance = nn.Sequential(*rnns)
        else:
            # 不使用复lstm
            self.enhance = nn.LSTM(
                input_size=hidden_dim*self.kernel_num[-1],
                hidden_size=self.rnn_units,
                num_layers=2,
                dropout=0.0,
                bidirectional=bidirectional,
                batch_first=False
            )
            self.tranform = nn.Linear(
                self.rnn_units * fac, hidden_dim*self.kernel_num[-1])
        self.linear = nn.Linear(self.ffn_input[0]*self.ffn_input[1]*kernel_num[-1], 2)
        self.avg = nn.AdaptiveAvgPool2d(self.ffn_input)

    
    def forward(self, inputs, lens=None):
        complex_data = self.get_amp_phase(inputs, lens)

        out = complex_data
        encoder_out = []
        print('before encoders: ', out.shape)

        # 通过Encoder
        for idx, layer in enumerate(self.encoder):
            out = layer(out)
        #    print('encoder', out.size())
            encoder_out.append(out)

        # out尺寸为四维: (B, C, D, L)
        print('after encoders: ', out.shape)
        batch_size, channels, dims, lengths = out.size()
        out = out.permute(3, 0, 1, 2) # 将out进行转置操作, out: (L, B, C, D)

        # 通过lstm
        print('before clstm: ', out.shape)
        if self.use_clstm:
            r_rnn_in = out[:, :, :channels//2]
            i_rnn_in = out[:, :, channels//2:]
            r_rnn_in = torch.reshape(
                r_rnn_in, [lengths, batch_size, channels//2*dims])
            i_rnn_in = torch.reshape(
                i_rnn_in, [lengths, batch_size, channels//2*dims])

            r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])

            r_rnn_in = torch.reshape(
                r_rnn_in, [lengths, batch_size, channels//2, dims])
            i_rnn_in = torch.reshape(
                i_rnn_in, [lengths, batch_size, channels//2, dims])
            out = torch.cat([r_rnn_in, i_rnn_in], 2)
            print('after clstm: ', out.shape)
        else:
            # to [L, B, C, D]
            out = torch.reshape(out, [lengths, batch_size, channels*dims]) # out: (L, B, C*D)
            out, _ = self.enhance(out)
            out = self.tranform(out)
            out = torch.reshape(out, [lengths, batch_size, channels, dims]) # out: (L, B, C, D)

        print('the final output: ', out.shape)
        out = out.permute(1, 2, 3, 0) # out: (B, C, D, L)
        print('after permute: ', out.shape)
        out = self.avg(out) # out: (B, C, ffn_input) == (B, C, D, L), ffn_input==(D, L)
        print('after avg: ', out.shape)
        out = out.view(out.shape[0], -1) # out: (B, C*D*L)
        print('before linear: ', out.shape)
        out = self.linear(out)
        return out.float()
    

    def get_amp_phase(self, inputs, lens=None):
        print('======== module dc_crn.py, DCCRN ========')
        print('inputs: ', inputs.shape)
        specs = self.stft(inputs) # specs尺寸为二维，这就是频域信号
        print('after ConvSTFT: ', specs.shape)
        real = specs[:, :self.fft_len//2+1]
        imag = specs[:, self.fft_len//2+1:]
        # 振幅信息
        spec_mags = torch.sqrt(real**2+imag**2+1e-8) # 振幅，即实部和虚部的平方和的算术平方根
        spec_mags = spec_mags
        print('振幅: ', spec_mags.shape)
        # 相位信息
        spec_phase = torch.atan2(imag, real) # 相位，即虚部与实部的比值的反正切函数值
        spec_phase = spec_phase
        print('相位: ', spec_phase.shape)

        cspecs = torch.stack([real, imag], 1)
        cspecs = cspecs[:, :, 1:]
        print('实部和虚部: ', cspecs.shape)
        return cspecs