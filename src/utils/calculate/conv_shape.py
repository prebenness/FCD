def calc_deconv_arch(in_features, out_shape, num_deconvs=3):
    '''
    Calculates the parameters of a deconvolution network for vector to
    image reconstruction
    Returns (
        in_channels, out_channels, kernel_size, stride, padding, output_padding
    )[]
    '''

    # TODO: implement properly
    # Debug - hard code known architecture
    assert num_deconvs == 3
    assert in_features == 512
    assert out_shape == (1, 28, 28)

    return [
        (32, 16, 4, 2, 1, 0),
        (16, 8, 3, 2, 2, 0),
        (8, 1, 4, 2, 0, 0),
    ]
