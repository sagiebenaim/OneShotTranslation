def create_model(opt):
    print(opt.model)
    if opt.model == 'ost':
        assert (opt.dataset_mode == 'unaligned')
        from .ost import OSTModel
        model = OSTModel()
    elif opt.model == 'autoencoder':
        assert (opt.dataset_mode == 'single')
        from .autoencoder_model import AutoEncoderModel
        model = AutoEncoderModel()
    elif opt.model == 'test':
        assert (opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
