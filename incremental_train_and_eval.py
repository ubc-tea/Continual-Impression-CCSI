#!/usr/bin/env python
# coding=utf-8
from utils_pytorch import *
from sklearn.metrics import confusion_matrix
import wandb
from models.layers.continual_normalization.cn_utils import *

cur_features = []
ref_features = []
old_scores = []
new_scores = []
def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs

def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs
    
    
    

def incremental_train_and_eval(ckpt_dir,log_f_name_train,log_f_name_val,epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
            trainloader, testloader, evalloader,\
            iteration, start_iteration, \
            lamda, \
            dist, K, lw_mr, CL,ro,\
            da_coef = 1,fix_bn=False, weight_per_class=None, device=None,alpha_3= 1,temprature = 5,continual_norm=False):
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    best_model = None

    if log_f_name_train != None:
        log_f_train = open(log_f_name_train,"w+")
        log_f_train.write('epoch,loss1,loss2,loss3,total_loss,acc\n')

        log_f_valid = open(log_f_name_val,"w+")
        log_f_valid.write('epoch,total_loss,acc\n')
    if iteration > start_iteration and ref_model != None :
        print("We have ref model")
        mask = torch.eye(tg_model.fc.fc1.out_features+tg_model.fc.fc2.out_features, dtype=torch.float32).to(device)
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features
        handle_ref_features = ref_model.fc.register_forward_hook(get_ref_features)
        handle_cur_features = tg_model.fc.register_forward_hook(get_cur_features)
        handle_old_scores_bs = tg_model.fc.fc1.register_forward_hook(get_old_scores_before_scale)
        handle_new_scores_bs = tg_model.fc.fc2.register_forward_hook(get_new_scores_before_scale)
    best_performance = 0
    for epoch in range(epochs):
        #train
        tg_model.train()
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    #m.weight.requires_grad = False
                    #m.bias.requires_grad = False
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        correct = 0
        total = 0
        if iteration > start_iteration and ref_model != None :
            s_centroids = torch.zeros(
                (tg_model.fc.fc1.out_features+tg_model.fc.fc2.out_features,tg_model.fc.fc1.out_features+tg_model.fc.fc2.out_features)).to(device)
            t_centroids = torch.zeros(
                (tg_model.fc.fc1.out_features+tg_model.fc.fc2.out_features,tg_model.fc.fc1.out_features+tg_model.fc.fc2.out_features)).to(device)
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        print('\nEpoch: %d, LR: ' % epoch, end='')
        print(tg_optimizer.param_groups[0]['lr'])
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            tg_optimizer.zero_grad()
            # print(tg_model)
            outputs = tg_model(inputs)
            # print('outpuuuuut',set(targets))
            if iteration == start_iteration or ref_model == None:
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            else:
                ref_outputs = ref_model(inputs)
                loss1 = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), \
                    torch.ones(inputs.shape[0]).to(device)) * lamda
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
                #################################################
                # scores before scale, [-1, 1]
                outputs_bs = torch.cat((old_scores, new_scores), dim=1)
                # print("outputs_bs",outputs_bs.shape)
                
                assert(outputs_bs.size()==outputs.size())
                #get groud truth scores
                gt_index = torch.zeros(outputs_bs.size()).to(device)
                # print("gt_index",gt_index.shape)
                gt_index = gt_index.scatter(1, targets.view(-1,1), 1).ge(0.5)
                # print("gt_index",gt_index.shape)
                gt_scores = outputs_bs.masked_select(gt_index)
                # print(gt_scores.shape)
                #get top-K scores on novel classes
                max_novel_scores = outputs_bs[:, num_old_classes:].topk(K, dim=1)[0]
                #the index of hard samples, i.e., samples of old classes
                hard_index = targets.lt(num_old_classes)
                hard_num = torch.nonzero(hard_index).size(0)
                # print("hard examples size: ", hard_num)
                if  hard_num > 0:
                    gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, K)
                    max_novel_scores = max_novel_scores[hard_index]
                    # print("gt_scores ",gt_scores)
                    # print("max_novel_scores ",max_novel_scores)
                    assert(gt_scores.size() == max_novel_scores.size())
                    assert(gt_scores.size(0) == hard_num)
                    #print("hard example gt scores: ", gt_scores.size(), gt_scores)
                    #print("hard example max novel scores: ", max_novel_scores.size(), max_novel_scores)
                    loss3 = nn.MarginRankingLoss(margin=dist)(gt_scores.view(-1, 1), \
                        max_novel_scores.view(-1, 1), torch.ones(hard_num*K,1).to(device)) * lw_mr
                else:
                    loss3 = torch.zeros(1).to(device)
                ################################################
                
                ####TODO LOSS4 Inter-Domain Contrastive Alignment
                
                # print(outputs_bs.shape)
                if iteration > start_iteration and ref_model != None:
                    s_features = torch.zeros(
                        (tg_model.fc.fc1.out_features+tg_model.fc.fc2.out_features,tg_model.fc.fc1.out_features+tg_model.fc.fc2.out_features)).to(device)



                    for i in range(len(s_features)):
                        indices = torch.where(targets==i)[0]
                        if len(indices) != 0:
                            s_features[i] = torch.sum(outputs_bs[indices],dim=0)/len(indices)
                    # print(s_features)

                    s_centroids = (1-ro)*s_centroids.detach() + ro*s_features
                    # print(s_centroids)

                    t_features = torch.zeros(
                        (tg_model.fc.fc1.out_features+tg_model.fc.fc2.out_features,tg_model.fc.fc1.out_features+tg_model.fc.fc2.out_features)).to(device)


                    t_inputs, t_targets = next(iter(evalloader))
                    # print("t_targets",t_targets)
                    t_inputs, t_targets = t_inputs.to(device), t_targets.to(device)
                    t_outputs = tg_model(t_inputs)
                    # print(t_outputs)
                    _, t_predicted = t_outputs.max(1)
                    t_outputs_bs = torch.cat((old_scores, new_scores), dim=1)

                    for i in range(len(t_features)):
                        indices = torch.where(t_predicted==i)[0]
                        if len(indices) != 0:
                            t_features[i] = torch.sum(
                                t_outputs_bs[indices],dim=0)/len(indices)
                    # loss4 = torch.sum(t_features)
    #                 print(t_outputs_bs.shape)
                    # loss4 = torch.sum(t_centroids)
                    # t_centroids.requires_grad = False
                    t_centroids = (1-ro)*t_centroids.detach() + ro*t_features
                    # print(t_centroids.requires_grad)
                    # loss4 = torch.sum(t_centroids)

                    t_centroids_norm = t_centroids / (t_centroids.norm(dim=1)[:, None]+1e-10)
                    s_centroids_norm = s_centroids / (s_centroids.norm(dim=1)[:, None]+1e-10)
                    res = torch.exp(torch.mm(t_centroids_norm, s_centroids_norm.transpose(0,1))/temprature)

                    loss4 = torch.sum(res)

                    # print(res,mask,torch.mul(res,mask))

                    loss4 = -1* torch.log(torch.sum(torch.diagonal(res,0)))+torch.log(torch.sum(res))
                        
                # print(loss4,torch.sum(torch.diagonal(res,0)),torch.sum(res))
                
            
                loss = loss1 + loss2 + alpha_3*loss3 + da_coef*loss4
                # loss = loss1 + loss2 
            loss.backward()
            tg_optimizer.step()

            train_loss += loss.item()
            if iteration > start_iteration and ref_model != None:
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()
                train_loss3 += loss3.item()
                train_loss4 += loss4.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #if iteration == 0:
            #    msg = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % \
            #    (train_loss/(batch_idx+1), 100.*correct/total, correct, total)
            #else:
            #    msg = 'Loss1: %.3f Loss2: %.3f Loss: %.3f | Acc: %.3f%% (%d/%d)' % \
            #    (loss1.item(), loss2.item(), train_loss/(batch_idx+1), 100.*correct/total, correct, total)
            #progress_bar(batch_idx, len(trainloader), msg)
        if iteration == start_iteration or ref_model == None:
            print('Train set: {}, Train Loss: {:.4f} Acc: {:.4f}'.format(\
                len(trainloader), train_loss/(batch_idx+1), 100.*correct/total))
            metrics = {"train accuracy": 100.*correct/total,
                       "train loss": train_loss/(batch_idx+1)}
            wandb.log(metrics)

        else:
            print('Train set: {}, Train Loss1: {:.4f}, Train Loss2: {:.4f}, Train Loss3: {:.4f}, Train Loss4: {:.4f},\
                Train Loss: {:.4f} Acc: {:.4f}'.format(len(trainloader), \
                train_loss1/(batch_idx+1), train_loss2/(batch_idx+1), train_loss3/(batch_idx+1),train_loss4/(batch_idx+1),
                train_loss/(batch_idx+1), 100.*correct/total))
            
            metrics = {"train loss 1": train_loss1/(batch_idx+1),
                       "train loss 2": train_loss2/(batch_idx+1),
                       "train loss 3": train_loss3/(batch_idx+1),
                       "train loss 4": train_loss4/(batch_idx+1),
                       "train loss": train_loss/(batch_idx+1),
                       "train accuracy": 100.*correct/total}
            wandb.log(metrics)
            
            # print('Train set: {}, Train Loss1: {:.4f}, Train Loss2: {:.4f},\
            #     Train Loss: {:.4f} Acc: {:.4f}'.format(len(trainloader), \
            #     train_loss1/(batch_idx+1), train_loss2/(batch_idx+1),
            #     train_loss/(batch_idx+1), 100.*correct/total))
        # ('epoch,loss1,loss2,loss3,total_loss,acc\n')
        # log_f_train.write('{},{},{},{},{},{}\n'.format(
        #     epoch,
        #     train_loss1/(batch_idx+1),
        #     train_loss2/(batch_idx+1),
        #     train_loss3/(batch_idx+1),
        #     train_loss/(batch_idx+1), 
        #     100.*correct/total))
        # log_f_train.flush() 

        #eval
        tg_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        num_classes = tg_model.fc.out_features
        # num_classes = 42
        cm = np.zeros((3, num_classes, num_classes))
        all_targets = []
        all_predicted = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                all_targets.append(targets.cpu())
                outputs = tg_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                all_predicted.append(predicted.cpu())
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                
        metrics = {"test loss": test_loss/(batch_idx+1),
                    "test accuracy":  100.*correct/total,
                  "Best Acc": best_performance}
        # print("test targets",set(all_targets))
        cm[0, :, :] = confusion_matrix(np.concatenate(all_targets), np.concatenate(all_predicted))
        class_num = len(cm[0, :, :])
        print(cm[0, :, :])
        true = cm[0, 0, 0] 
        all_ = np.sum(cm[0, 0, :])
        for i in range(0,class_num):
            
            true += cm[0, i, i]
            all_ += np.sum(cm[0, i, :])
            print("ACC "+str(i+1)+": "+ str(true/all_))
            metrics["ACC "+str(i+1)] = true/all_
        acc_last = cm[0, class_num-1 , class_num-1]/np.sum(cm[0, class_num-1 , :])
        print("ACC on last class: " + str(acc_last))
        
        print('Test set: {} Test Loss: {:.4f} Acc: {:.4f}'.format(\
            len(testloader), test_loss/(batch_idx+1), 100.*correct/total))
        # log_f_valid.write('{},{},{}\n'.format(
        #     epoch,
        #     test_loss/(batch_idx+1), 
        #     100.*correct/total))
        # log_f_valid.flush() 


    
        wandb.log(metrics)
        
        tg_lr_scheduler.step(correct/total)
        if 100.*correct/total >= best_performance:
            best_performance = 100.*correct/total
            best_model = tg_model 
            # Save the model checkpoint
            group_running_mean_list = []
            group_running_var_list = []
            b_size_list = []
            print("saving_best")
            if continual_norm:
                group_running_mean_list,group_running_var_list,b_size_list = get_continual_variables(tg_model,'model')
                print('lens: ',len(group_running_mean_list),len(group_running_var_list),len(b_size_list))
            torch.save({
                'e.poch': epoch,
                'model_state_dict': tg_model.state_dict(),
                'optimizer_state_dict': tg_model.state_dict(),
                'loss': test_loss/(batch_idx+1),
                'Acc': best_performance, 
                'group_running_mean_list':group_running_mean_list,
                'group_running_var_list':group_running_var_list,
                'b_size_list': b_size_list},
                ckpt_dir)

    if iteration > start_iteration and ref_model != None:
        print("Removing register_forward_hook")
        handle_ref_features.remove()
        handle_cur_features.remove()
        handle_old_scores_bs.remove()
        handle_new_scores_bs.remove()
    return best_model