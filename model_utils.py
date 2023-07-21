import torch
import torch.nn.functional as F
import os

def gnn_pre_train(model, loader, optimizer, device, multi=True, loss_type="exp"):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch, embedding=False)
        proxy = data.proxys.float().reshape(output.shape[0], -1)
        loss = F.mse_loss(output, proxy)

        loss.backward()
        optimizer.step()
        total_loss += float(loss) * len(data)

    return total_loss / len(loader)

@torch.no_grad()
def gnn_pre_eval(model, loader, device, multi=True, loss_type="exp"):
    model.eval()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch, embedding=False)
        proxy = data.proxys.float().reshape(output.shape[0], -1)
        loss = F.mse_loss(output, proxy)

        total_loss += float(loss) * len(data)

    return total_loss

def gnn_train(model, loader, optimizer, device, multi=True, loss_type="exp"):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch, embedding=False)
        y = data.y.float().reshape(output.shape[0], -1)
        if not multi:
            y = y[:, -1]
            y = y.reshape(-1, 1)
        if loss_type == "exp":
            loss = F.mse_loss(torch.exp(output - 1), torch.exp(y - 1))
        else:
            loss = F.mse_loss(output, y)

        loss.backward()
        optimizer.step()
        total_loss += float(loss) * len(data)

    return total_loss / len(loader)

@torch.no_grad()
def gnn_eval(model, loader, device, multi=True, loss_type="exp"):
    model.eval()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch, embedding=False)
        y = data.y.float().reshape(output.shape[0], -1)
        if not multi:
            y = y[:, -1]
            y = y.reshape(-1, 1)
        if loss_type == "exp":
            loss = F.mse_loss(torch.exp(output - 1), torch.exp(y - 1))
        elif loss_type == "mse":
            loss = F.mse_loss(output, y)
        elif loss_type == "mae":
            loss = F.l1_loss(output, y)

        total_loss += float(loss) * len(data)

    return total_loss


def gnn_ranking_train(model, loader, optimizer, device, multi=True, loss_type="exp"):
    model.train()
    total_loss = 0
    for data in loader:
        data1, data2 = data
        data1 = data1.to(device)
        data2 = data2.to(device)
        
        optimizer.zero_grad()
        output1 = model(data1.x, data1.edge_index, data1.batch, embedding=False)
        output2 = model(data2.x, data2.edge_index, data2.batch, embedding=False)

        scalar = torch.Tensor([0.1 for _ in range(output2.shape[0])]).to(device)
        output2 = torch.add(output2, scalar)

        loss = torch.mean(torch.max(torch.zeros(output1.shape).to(device), output2-output1))

        loss.backward()
        optimizer.step()
        total_loss += float(loss) * len(data)

    return total_loss / len(loader)

@torch.no_grad()
def gnn_ranking_eval(model, loader, device, multi=True, loss_type="exp"):
    model.eval()
    total_loss = 0

    for data in loader:
        data1, data2 = data
        data1 = data1.to(device)
        data2 = data2.to(device)

        output1 = model(data1.x, data1.edge_index, data1.batch, embedding=False)
        output2 = model(data2.x, data2.edge_index, data2.batch, embedding=False)

        scalar = torch.Tensor([0.1 for _ in range(output2.shape[0])]).to(device)
        output2 = torch.add(output2, scalar)

        loss = torch.mean(torch.max(torch.zeros(output1.shape).to(device), output2-output1))
        total_loss += float(loss) * len(data)

    return total_loss / len(loader)

# class AverageMeter(object):
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.avg = 0
#         self.sum = 0
#         self.cnt = 0

#     def update(self, val, n=1):
#         self.sum += val * n
#         self.cnt += n
#         self.avg = self.sum / self.cnt


# class ControllerDataset(torch.utils.data.Dataset):
#     def __init__(self, inputs, targets=None, train=True, sos_id=0, eos_id=0):
#         super(ControllerDataset, self).__init__()
#         if targets is not None:
#             assert len(inputs) == len(targets)
#         self.inputs = inputs
#         self.targets = targets
#         self.train = train
#         self.sos_id = sos_id
#         self.eos_id = eos_id

#     def __getitem__(self, index):
#         encoder_input = self.inputs[index]
#         encoder_target = None
#         if self.targets is not None:
#             encoder_target = [self.targets[index]]
#         if self.train:
#             decoder_input = [self.sos_id] + encoder_input[:-1]
#             sample = {
#                 "encoder_input": torch.LongTensor(encoder_input),
#                 "encoder_target": torch.FloatTensor(encoder_target),
#                 "decoder_input": torch.LongTensor(decoder_input),
#                 "decoder_target": torch.LongTensor(encoder_input),
#             }
#         else:
#             sample = {
#                 "encoder_input": torch.LongTensor(encoder_input),
#                 "decoder_target": torch.LongTensor(encoder_input),
#             }
#             if encoder_target is not None:
#                 sample["encoder_target"] = torch.FloatTensor(encoder_target)
#         return sample

#     def __len__(self):
#         return len(self.inputs)


# def controller_train(train_queue, model, optimizer, params, device, exp=False):

#     objs = AverageMeter()
#     mse = AverageMeter()
#     nll = AverageMeter()
#     model.train()
#     for step, sample in enumerate(train_queue):

#         encoder_input = sample["encoder_input"].to(device)
#         encoder_target = sample["encoder_target"].to(device)
#         decoder_input = sample["decoder_input"].to(device)
#         decoder_target = sample["decoder_target"].to(device)

#         optimizer.zero_grad()
#         predict_value, log_prob, arch = model(encoder_input, decoder_input)
#         if exp:
#             loss_1 = F.mse_loss(torch.exp(predict_value.squeeze() - 1),
#                                 torch.exp(encoder_target.squeeze() - 1))
#         else:
#             loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze())
#         loss_2 = F.nll_loss(
#             log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1)
#         )
#         loss = params.trade_off * loss_1 + (1 - params.trade_off) * loss_2
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_bound)
#         optimizer.step()

#         n = encoder_input.size(0)
#         objs.update(loss.data, n)
#         mse.update(loss_1.data, n)
#         nll.update(loss_2.data, n)

#     return objs.avg, mse.avg, nll.avg


# def controller_infer(queue, model, step, device, direction="+"):
#     new_arch_list = []
#     new_predict_values = []
#     model.eval()
#     for i, sample in enumerate(queue):
#         encoder_input = sample["encoder_input"].to(device)
#         model.zero_grad()
#         new_arch, new_predict_value = model.generate_new_arch(
#             encoder_input, step, direction=direction
#         )
#         new_arch_list.extend(new_arch.data.squeeze().tolist())
#         new_predict_values.extend(new_predict_value.data.squeeze().tolist())
#     return new_arch_list, new_predict_values


# def train_controller(model, train_input, train_target, epochs, params, device, exp=False):
#     controller_train_dataset = ControllerDataset(train_input, train_target, True)
#     controller_train_queue = torch.utils.data.DataLoader(
#         controller_train_dataset,
#         batch_size=params.batch_size,
#         shuffle=True,
#         pin_memory=True,
#         drop_last=False,
#     )
#     optimizer = torch.optim.Adam(model.parameters(), lr=params.lr,
#                                  weight_decay=params.weight_decay)

#     loss_hist = []
#     for epoch in range(1, epochs + 1):
#         loss, mse, ce = controller_train(controller_train_queue, model,
#                                          optimizer, params, device, exp)
#         loss_hist.append(loss)

#     return loss_hist