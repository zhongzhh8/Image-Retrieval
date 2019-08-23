import argparse
from model import *
from optim import get_optimizer
from utils import *
from datasets import CUB_200,Stanford_Dogs


def get_parser():
    parser = argparse.ArgumentParser(description='train DSH')

    parser.add_argument('--dataset_name', default='CUB', help='CUB or Stanford_Dogs')
    parser.add_argument('--binary_bits', type=int, default=64, help='length of hashing binary')
    parser.add_argument('--margin', type=float, default=20, help='loss_type')  # 取bit的四分之一，margin影响很大
    parser.add_argument('--lr', type=float, default=0.001, help='lr=0.001')
    parser.add_argument('--ngpu', type=int, default=2, help='which GPU to use')

    parser.add_argument('--load_model', default=False, help='wether load model checkpoints or not')
    parser.add_argument('--load_path', default='/home/disk3/a_zhongzhanhui/PycharmProject/image_retrieval_mine/checkpoints/Stanford_Dogs_SGD_64bits_20margin_/triplet_resnet_64bits_400.pth', help='location to load model')
    parser.add_argument('--optimizer_name', default='SGD', help='SGD or adam_inverse_sqrt')
    parser.add_argument('--batchSize', type=int, default=80, help='input batch size')
    parser.add_argument('--log_postfix', default='', help='without_if_database')

    parser.add_argument('--outf', default='./checkpoints/', help='')
    parser.add_argument('--num_epochs', type=int, default=4001, help='number of epochs to train for')
    parser.add_argument('--step_lr', type=int, default=1800, help='change lr per strp_lr epoch')
    parser.add_argument('--checkpoint_step', type=int, default=50, help='checkpointing after batches')
    parser.add_argument('--datapath_CUB', default='/home/disk3/a_zhongzhanhui/data/CUB_200_2011',
                        help='data path of CUB')
    parser.add_argument('--datapath_dog', default='/home/disk3/a_zhongzhanhui/data/Stanford_Dogs',
                        help='data path of Stanford_Dogs')

    return parser




def triplet_hashing_loss_regu(embeddings, labels, margin,opt):
    triplets = get_triplets(labels)

    if embeddings.is_cuda:
        triplets = triplets.cuda(opt.ngpu)

    anchor_embeddings=embeddings[triplets[:, 0]]
    positive_embeddings=embeddings[triplets[:, 1]]
    negative_embeddings=embeddings[triplets[:, 2]]

    ap_distances = (anchor_embeddings - positive_embeddings).pow(2).sum(1)  # 计算anchor与positive之间的距离
    an_distances = (anchor_embeddings - negative_embeddings).pow(2).sum(1)  #  计算anchor与negative之间的距离

    losses = F.relu(ap_distances - an_distances + margin)

    return losses.mean()

    # triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    # output = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
    # return output

@timing
def compute_result(dataloader, net, opt):
    all_codes, all_labels = [], []
    net.eval() #不使用BN和dropout
    sub = (torch.zeros(opt.binary_bits))
    for imgs, labels in dataloader:
        codes= net(Variable(imgs.cuda(opt.ngpu), volatile=True))
        all_codes.append((codes.data.cpu())-sub)
        all_labels.append(labels)

    return torch.sign(torch.cat(all_codes)), torch.cat(all_labels)



def get_dataset(DataSet,batchSize,datapath):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),  # 训练集才需要做这一步处理，获得更多的随机化数据
        transforms.ToTensor(),
        normalize])

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize])

    testset = DataSet(root=datapath, train=False,transform_train=transform_train, transform_test=transform_test)
    test_loader = torchdata.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=2, pin_memory=True)

    train_set = DataSet(root=datapath,train=True, transform_train=transform_train, transform_test=transform_test)
    train_loader_ = torchdata.DataLoader(train_set, batch_size=batchSize, shuffle=True, num_workers=2,pin_memory=True)  # 暂时的loader
    labels = []
    for batch_idx, (data, target) in enumerate(train_loader_):
        labels.extend(target)
    if DataSet==CUB_200:
        cRS=categoryRandomSampler_CUB
    elif DataSet==Stanford_Dogs:
        cRS=categoryRandomSampler_dog
    casampler = cRS(numBatchCategory=10, targets=labels,batch_size=batchSize)  # 保证batch里面至少有10个类别，This sampler will sample numBatchCategory categories in each batch.
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=False, num_workers=2,sampler=casampler)

    DB_set = DataSet(root=datapath,train=True, transform_train=transform_train, transform_test=transform_test)
    DB_loader = torchdata.DataLoader(DB_set, batch_size=batchSize, shuffle=True, num_workers=2, pin_memory=True)
    #训练的时候用train_loader，计算mAP的时候用DBloader和test_loader

    return train_loader,DB_loader,test_loader




def main():
    parser=get_parser()
    opt = parser.parse_args()
    log_name = opt.dataset_name +'_'+opt.optimizer_name+ '_' + str(opt.binary_bits) + 'bits_' + str(opt.margin)+'margin_'+opt.log_postfix
    opt.outf=opt.outf+log_name
    os.makedirs(opt.outf, exist_ok=True)
    print(opt)

    feed_random_seed()
    logger = logging.getLogger()
    set_logger(logger, log_name)
    logger.info(opt)  # 写进日志文件里


    print('====loading data====')
    load_time_start = time.time()
    if opt.dataset_name=='CUB':
        dathpath=opt.datapath_CUB
        DataSetClass=CUB_200
    elif opt.dataset_name=='Stanford_Dogs':
        dathpath = opt.datapath_dog
        DataSetClass=Stanford_Dogs
    train_loader, DB_loader, test_loader = get_dataset(DataSetClass, opt.batchSize, dathpath)
    load_time_end = time.time()
    print( load_time_end - load_time_start,' s')
    print('====data loaded====')


    net =  HashModel(opt.binary_bits)

    if opt.load_model==True:
        net.load(opt.load_path)

    device = torch.device("cuda:"+str(opt.ngpu) if torch.cuda.is_available() else "cpu")
    net.to(device)
    if opt.optimizer_name=='SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005)
    elif opt.optimizer_name=='adam_inverse_sqrt':
        optimizer = get_optimizer(net.parameters(), 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.001')
    else:
        print('No optimizer!=========')

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_lr, gamma=0.1)

    max_map = 0
    print('===start training===')
    epoch_avg_loss=0
    for epoch in range(0, opt.num_epochs):
        total_loss = 0
        net.train()
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = [Variable(x.cuda(opt.ngpu)) for x in (imgs, labels)]
            net.zero_grad()
            img_embeddings = net(imgs)
            triplet_loss = triplet_hashing_loss_regu(img_embeddings, labels, opt.margin, opt)
            triplet_loss.backward()
            optimizer.step()  # 每个mini-batch里面，只有用了optimizer.step()，模型才会更新
            total_loss += triplet_loss

        scheduler.step()  # 每个epoch里用一次，scheduler.step()是对lr进行调整，经过step_size次调用后更新一次lr（变为0.1倍）

        epoch_loss= total_loss / len(train_loader)
        print(f'epoch: {epoch:03d} epoch_loss: {epoch_loss:.4f}', end='\n')
        epoch_avg_loss+=epoch_loss/opt.checkpoint_step

        if epoch > 1 and (epoch % opt.checkpoint_step == 0 or epoch ==opt.num_epochs-1):
            # compute mAP by searching testset images from trainset
            train_binary, train_label = compute_result(DB_loader, net, opt)
            test_binary, test_label = compute_result(test_loader, net, opt)
            mAP = compute_mAP(train_binary, test_binary, train_label, test_label)
            max_map = max(mAP, max_map)

            print(f'--------------------[{epoch}] avg_loss: {epoch_avg_loss:.4f} retrieval mAP: {mAP:.4f} max mAP: {max_map:.4f}')
            logger.info(f'[{epoch}] avg_loss: {epoch_avg_loss:.4f} retrival map {mAP:.4f} max_map {max_map:.4f}')  #写进日志文件里

            epoch_avg_loss=0
            # save checkpoints
            print('save checkpoints')
            save_path=os.path.join(opt.outf, f'triplet_resnet_{opt.binary_bits}bits_{epoch:03d}.pth')
            net.save(save_path)

    return max_map, opt


if __name__ == '__main__':
    max_map, opt = main()
    print(opt)
    print('retrieval mAP: ', max_map)

