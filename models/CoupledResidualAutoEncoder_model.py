import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from losses.EdgeLoss import EdgeLoss
from losses.TextureLoss import TextureLoss

class Encoder(nn.Module):
    def __init__(self, hidden_dims, in_channels=3, encoded_dim=256):
        super().__init__()

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        self.relu = nn.ReLU(inplace=True)

        # self.lin1 = nn.Linear(hidden_dims[-1]*4, 128)
        # self.lin2 = nn.Linear(128, encoded_dim)
        self.lin1 = nn.Linear(hidden_dims[-1]*4, encoded_dim)

    def forward(self, x):
        outs = []
        for seq in self.encoder:
            x = seq(x)
            outs.append(x)
        x = torch.flatten(x, start_dim=1)

        x = self.lin1(x)
        x = self.relu(x)
        # x = self.lin2(x)

        return x, outs

class DAEncoder(nn.Module):
    def __init__(self, hidden_dims, in_channels=3, encoded_dim=256):
        super().__init__()

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        self.relu = nn.ReLU(inplace=True)

        # self.lin1 = nn.Linear(hidden_dims[-1]*4, 128)
        # self.lin2 = nn.Linear(128, encoded_dim)
        self.lin1 = nn.Linear(hidden_dims[-1]*4, encoded_dim)

    def forward(self, x, src_outs):
        outs = []
        count = -1
        for seq in self.encoder:
            x = seq(x)
            for i in range(len(outs)):
                outs[i] = seq(outs[i])
            if (count >= 0):
                outs.append(seq(src_outs[count]))
            count += 1
        x = torch.flatten(x, start_dim=1)
        for i in range(len(outs)):
            outs[i] = torch.flatten(outs[i], start_dim=1)

        x = self.lin1(x)
        x = self.relu(x)
        for i in range(len(outs)):
            outs[i] = self.lin1(outs[i])
            outs[i] = self.relu(outs[i])
        # x = self.lin2(x)

        return x, outs

class Decoder(nn.Module):
    def __init__(self, hidden_dims, encoded_dim=256):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # self.lin1 = nn.Linear(encoded_dim, 128)
        # self.lin2 = nn.Linear(128, hidden_dims[-1]*4)
        self.lin1 = nn.Linear(encoded_dim, hidden_dims[-1]*4)

        modules = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.output_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        # x = self.lin2(x)
        # x = self.relu(x)

        x = x.view(-1, 512, 2, 2)
        x = self.decoder(x)
        x = self.output_layer(x)

        return x

#PyTorch defined model
class CoupledResidualAutoEncoder(nn.Module):
    """basenet for fer2013"""
    def __init__(self, encoded_dim=256):
        super(CoupledResidualAutoEncoder, self).__init__()

        hidden_dims = [128, 256, 512]

        self.encoder = Encoder(hidden_dims, 3, encoded_dim)
        self.decoder = Decoder(hidden_dims, encoded_dim)

        self.lin = nn.Linear(2*encoded_dim, encoded_dim)
        self.relu = nn.ReLU()

        hidden_dims.reverse()
        self.da_encoder = Encoder(hidden_dims, 3, encoded_dim)
        self.trg_decoder = Decoder(hidden_dims, encoded_dim)

    def forward(self, x, x_trg):
        x_src, enc_outs = self.encoder.forward(x)
        x_trg, da_enc_outs = self.da_encoder.forward(x_trg)

        da_out = []
        da_out.append(x_src)
        da_out.append(x_trg)
        da_out = torch.cat(da_out, 1)

        x_da = self.lin(da_out)
        x_da = self.relu(x_da)

        x_src = self.decoder.forward(x_src)
        x_da_src = self.decoder.forward(x_da)
        x_trg = self.trg_decoder.forward(x_trg)
        x_da_trg = self.trg_decoder.forward(x_da)

        return x_src, x_da_src, x_trg, x_da_trg

#The abstract model class, uses above defined class and is used in the train script
class CoupledResidualAutoEncodermodel(BaseModel):
    """basenet for fer2013"""

    def __init__(self, configuration):
        super().__init__(configuration)

        #Initialize model defined above
        self.model = CoupledResidualAutoEncoder(configuration['encoded_dim'])
        self.model.cuda()

        self.gamma = configuration['gamma']

        #Define loss function
        self.criterion_loss = nn.MSELoss().cuda()
        self.edge_loss = EdgeLoss().cuda()
        self.texture_loss = TextureLoss().cuda()
        #Define optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=configuration['lr'],
            betas=(configuration['momentum'], 0.999),
            weight_decay=configuration['weight_decay']
        )

        #Need to include these arrays with the optimizers and names of loss functions and models
        #Will be used by other functions for saving/loading
        # self.optimizers = [self.optimizers[i] for i in range(4)]
        self.optimizers = [self.optimizer]
        self.loss_names = ['total']
        self.network_names = ['model']

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []

    #Calls the models forwards function
    def forward(self):
        x_src = self.source
        x_trg = self.target
        self.output_src, self.output_da_src, self.output_trg, self.output_da_trg = self.model.forward(x_src, x_trg)
        return self.output_src, self.output_da_src, self.output_trg, self.output_da_trg

    #Computes the loss with the specified name (in this case 'total')
    def compute_loss(self):
        # self.loss_edge = self.edge_loss(self.output_da_src, self.source) + self.edge_loss(self.output_da_trg, self.target)
        # self.loss_mse_recon = self.criterion_loss(self.output_src, self.source) + self.criterion_loss(self.output_trg, self.target)
        # self.loss_mse_da = self.criterion_loss(self.output_da_src, self.source) + self.criterion_loss(self.output_da_trg, self.target)
        # self.loss_mse_da = self.criterion_loss(self.output_da_trg, self.output_da_src)
        # self.loss_mse = self.loss_mse_recon# + self.loss_mse_da
        # self.loss_total = (1-self.gamma)*self.loss_edge + (self.gamma)*(self.loss_mse)


        self.loss_texture = self.texture_loss(self.output_src, self.source) + self.texture_loss(self.output_trg, self.target)
        self.loss_edge = self.edge_loss(self.output_src, self.source) + self.edge_loss(self.output_trg, self.target)
        self.loss_mse = self.criterion_loss(self.output_src, self.source) + self.criterion_loss(self.output_trg, self.target)
        self.loss_total = (1-self.gamma)*self.loss_edge + (1-self.gamma)*self.loss_texture + self.gamma*self.loss_mse

    #Compute backpropogation for the model
    def optimize_parameters(self):
        self.loss_total.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        torch.cuda.empty_cache()

    #Test function for the model
    def test(self):
        super().test() # run the forward pass

        # save predictions and labels as flat tensors
        # self.val_images.append(self.input)
        # self.val_predictions.append(self.output)
        # self.val_labels.append(self.label)

    #Should be run after each epoch, outputs accuracy
    def post_epoch_callback(self, epoch, visualizer):
        self.val_predictions = torch.cat(self.val_predictions, dim=0)
        predictions = torch.argmax(self.val_predictions, dim=1)
        predictions = torch.flatten(predictions).cpu()

        self.val_labels = torch.cat(self.val_labels, dim=0)
        labels = torch.flatten(self.val_labels).cpu()

        self.val_images = torch.squeeze(torch.cat(self.val_images, dim=0)).cpu()

        # Calculate and show accuracy
        val_accuracy = accuracy_score(labels, predictions)

        metrics = OrderedDict()
        metrics['Accuracy'] = val_accuracy

        if (visualizer != None):
            visualizer.plot_current_validation_metrics(epoch, metrics)
        print('Validation accuracy: {0:.3f}'.format(val_accuracy))

        # Here you may do something else with the validation data such as
        # displaying the validation images or calculating the ROC curve

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []

    def load_autoencoder(self, weights):
        keys = list(weights.keys())
        with torch.no_grad():
            for key in keys:
                var_list = key.split('.')
                layer = self.model
                for v in var_list:
                    layer = getattr(layer, v)
                layer.copy_(weights[key])

        for param in self.model.encoder.parameters():
            param.requires_grad = False
        for param in self.model.decoder.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    net = TEMPLATEmodel().cuda()
    from torchsummary import summary

    print(summary(net, input_size=(1, 48, 48)))
