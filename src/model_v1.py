from baseline import *

config["model"]["conv_0"] = {
    "params": {
        "in_channels": config["model"]["dim_feature"],
        "out_channels": 128,
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "bias": True
    }
}
config["model"]["conv_1"] = {
    "params": {
        "in_channels": 128,
        "out_channels": config["model"]["num_class"],
        "kernel_size": 3,
        "stride": 1,
        "padding": 0,
        "bias": True
    }
}

class PeModelV1(PeModel):
    def __init__(self, config):
        super().__init__(config)

        # const
        self.conv1d_0 = nn.Conv1d(
            **self.config["conv_0"]["params"]
        )
        self.conv1d_1 = nn.Conv1d(
            **self.config["conv_1"]["params"]
        )

    def forward(self, ids, masks):
        out = self.base_model(ids, masks)
        out = self.dropout(out[0])
        out = out.permute(0, 2, 1)
        out = self.conv1d_0(out)
        out = F.relu(out)
        out = self.conv1d_1(out)
        out, _ = torch.max(out, dim=2)
        return out
