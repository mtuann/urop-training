import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
from torchvision import datasets, transforms
from torchvision.models import resnet18
import copy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FLServer:
    def __init__(self) -> None:

        self.clients = []

    def add_client(self, fl_client):
        self.clients.append(fl_client)
        # print(f"Done adding new client with id: {fl_client.id}")
        pass

    def test(self, model):
        # print accuracy of model on each client
        test_metrics = []
        for flclient in self.clients:
            test_metric = flclient.test(model)
            test_metrics.append(test_metric)
        print(test_metrics)
    



class FLClient:

    def __init__(self, id, train_data, test_data):
        self.id = id
        self.train_data = train_data
        self.test_data = test_data
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=256, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=256, shuffle=False)
        
    def training(self, model, num_epoch=2):
        
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        model.train()
        for epoch in range(num_epoch):
            total_loss, correct, num_sample = 0, 0, 0
            for idb, (data, target) in enumerate(self.train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)

                loss = F.cross_entropy(output, target)
                total_loss += loss.item() * len(data)
                num_sample += len(data)
                
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                loss.backward()
                optimizer.step()
            # print(f"At epoch {epoch} correct: {correct} / {num_sample} loss: {total_loss:.4f} acc: {(100. * correct/ num_sample):.4f}")

        total_loss /= num_sample
        accuracy = 100. * correct / num_sample
        
        return model.state_dict(), {'loss': total_loss, 'accuracy': accuracy, '#data': num_sample, 'correct': correct}

    def test(self, model):

        model.eval()
        total_loss, correct, num_sample = 0, 0, 0
        for _, (data, target) in enumerate(self.test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            loss = F.cross_entropy(output, target)
            total_loss += loss.item() * len(data)
            num_sample += len(data)
            
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        return {'id': self.id, 'loss': total_loss / num_sample, 'accuracy': 100. * correct / num_sample, '#data': num_sample, 'correct': correct}




class Helper:
    def __init__(self) -> None:
        self.load_data()

    def load_data(self):
        # Load data from torch
        # List of dataset: MNIST, CIFAR-10, Fashion-MNIST
        # CIFAR-10 dataset
        transform = transforms.Compose([ transforms.ToTensor(), 
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) ) ])

        
        train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

        print(len(train_data), len(test_data))

        # statistic about number of samples in each class
        # train_stat = {}
        # for data, label in train_data: # list: [[data, label], ...]
        #     if label not in train_stat:
        #         train_stat[label] = 0
        #     train_stat[label] += 1
        # print(f"Training stat: {train_stat}")

        # test_stat = {}

        # for data, label in test_data: # list: [[data, label], ...]
        #     if label not in test_stat:
        #         test_stat[label] = 0
        #     test_stat[label] += 1
        # print(f"Test stat: {test_stat}")

        # ds_classes = {}
        # for i, (_, label) in enumerate(test_data):
        #     ds_classes.setdefault(label, []).append(i)
        # print(f"ds_classes: {ds_classes}")
        # import IPython; IPython.embed()

        # train_data, test_data
        # FL: divided data -> N clients
        # non-IID, IID
        # IID: N client, K = 50.000 / N; [0..k-1], [k..2k-1], [2k..3k-1],.... [..49999]
        # Define FL settings
        num_clients = 10
        client_train_data = torch.utils.data.random_split(train_data, [len(train_data) // num_clients] * num_clients)
        client_test_data = torch.utils.data.random_split(test_data, [len(test_data) // num_clients] * num_clients)
        
        # import IPython; IPython.embed()
        # stat data of each client
        # for idc in range(num_clients):
        #     data_client = client_train_data[idc]
        #     stat_client = {}
        #     for data, label in data_client: # list: [[data, label], ...]
        #         if label not in stat_client:
        #             stat_client[label] = 0
        #         stat_client[label] += 1
        #     print(f"Client number: {idc} with data: {stat_client}")
        
        # FL protocol: sever -> client: global model
        # for each epoch: client receives global model; training -> local model
        # client -> server: local model
        # sever: aggregation = Avg(weight num_client model); [ [1000 elements], .... [1000 elements]] -> [1000 elements]



        # exit(0)        
        flsever = FLServer()
        for idc in range(num_clients):
            client = FLClient(idc, client_train_data[idc], client_test_data[idc])
            flsever.add_client(client)
        

        
        global_model = resnet18(weights=None, num_classes=10).to(device)

        num_com_round = 10
        for id_com_round in range(num_com_round):
            print(f"Training at communication round id: {id_com_round}")
            list_local_models = []

            global_weights = copy.deepcopy(global_model.state_dict())

            for key in global_weights.keys():
                global_weights[key] = torch.zeros_like(global_weights[key]).to(torch.float32)

            for flclient in flsever.clients:
                # print(f"Start training client number: {flclient.id}")
                client_weights, train_metrics = flclient.training(copy.deepcopy(global_model))
                print(f"Finish training client number: {flclient.id} with metrics: {train_metrics}")
                # list_local_models.append(local_model_dict)
                # import IPython; IPython.embed(); exit(0)

                for _, key in enumerate(global_weights.keys()):
                    update_key = client_weights[key] / num_clients
                    global_weights[key].add_(update_key)

            global_model.load_state_dict(global_weights)

            flsever.test(global_model)
            
            print("**"*50)
            


if __name__ == '__main__':
    # set fixed random seed
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # random.seed(random_seed)
    
    helper = Helper()
