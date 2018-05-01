class config:
    def __init__(self, dataset):
        if dataset == "fashion-mnist" or dataset == "mnist":
            self.y_dim = 10
            self.image_shape = [28, 28, 1]
            self.c_dim = 1
            self.z_dim = 2
            self.f_dim = 64
            self.fc_dim = 1024
            self.lamb = 1e-2
            self.mem_size = 4096
            self.choose_k = 128
            self.key_dim = 256

        elif dataset == "cifar10":
            self.y_dim = 10
            self.image_shape = [32, 32, 3]
            self.c_dim = 3
            self.z_dim = 16
            self.f_dim = 128
            self.fc_dim = 1024
            self.lamb = 1e-6
            self.mem_size = 16384
            self.choose_k = 256
            self.key_dim = 512
