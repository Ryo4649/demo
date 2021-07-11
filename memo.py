# -*- coding: utf-8 -*-
import csv
import os
from glob import *
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from keras.utils import to_categorical

import sys
import warnings

from multiprocessing import Pool
import multiprocessing as multi
from contextlib import closing

# 指定した列のファイルデータを返す
def Read_csv_columns(path,columns,user=None):
    df = pd.read_csv(path)
    # userが指定されるとその被験者のみ参照
    if user != None:
        df = df[df["user_id"] == user]
    target_columns = df[columns].values
    return target_columns

# csvファイルのパスを渡すとファイルデータの配列を返す
def Read_csv(path):
    data = []
    with open(path,'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            data.append(r)
    return data

# 正解ラベルと予測ラベルを渡すとaccuracyを返す
def Calc_accuracy(y_true,y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

# 正解ラベルの配列を渡すとラベル1の割合を返す
def Label1_num(y_true):
    y_true[y_true!=0]=1
    s_bool = y_true == 1
    label1_num = s_bool.sum()
    label1_rate = float(label1_num)/len(y_true)
    return label1_rate
    # 実行例
    # path = "../dataset/dataset_base.csv"
    # df=pd.read_csv(path)
    # l=df['label'].values
    # print(Label1_num(l))

# pr曲線作成
def Mk_pr():
    # data_path = "../processed/all_10select/merge/merge.csv"
    # data_paths = ["../processed_haiowaei/all_all_select/merge/merge.csv", "../processed_haiowaei_baseline/all_all_select/merge/merge.csv", "../processed_minfreq/all_all_select/merge/merge.csv","../allpred1.csv","../allpred-1.csv"]
    data_paths = ["../processed_haiowaei/all_all_select/merge/merge.csv", "../processed_haiowaei_baseline/all_all_select/merge/merge.csv", "../processed_minfreq/all_all_select/merge/merge.csv","../allpred1.csv"]
    result_dir_path = "../all"
    if not os.path.exists(result_dir_path):
        os.makedirs(result_dir_path)

    # 個別に全員したいとき
    # users = []
    # for i in range(30):
    #     n = str(i)
    #     n = n.zfill(2)
    #     x = 'P'+n
    #     users.append(x)
    
    all_pr_auc = []
    ispe = 0
    for data_path in data_paths:
        # if u == 'P00': continue
        # if u == 'P22': continue
        # if u == 'P25': continue
        # if u == 'P29': continue
        # print(u)

        y_true = Read_csv_columns(data_path,'y_true')
        y_true[y_true!=0]=1
        y_proba = Read_csv_columns(data_path,'y_proba')

        precision,recall,thresholds = precision_recall_curve(y_true, y_proba)
        

        print(recall)
        print(precision)
        print(thresholds)

        print("precision: {} -- {}, recall: {} -- {}, threshold: {} -- {}".format(precision[0], precision[-1],recall[0], recall[-1],thresholds[0], thresholds[-1]))
        #area = auc(recall, precision)
        #settings = filename.split("_")


        precision_intpl = []
        recall_intpl = []
        # make interpolated curve
        for i in range(11):
            irecall = float(i) / 10.0
            max_prec = 0.0
            for ix, x in enumerate(recall):
                if x < irecall:
                    continue
                if precision[ix] > max_prec:
                    max_prec = precision[ix]
            precision_intpl += [max_prec]
            recall_intpl += [irecall]

        pr_auc = auc(recall_intpl,precision_intpl)
        print(pr_auc)
        # all_pr_auc += [[u,pr_auc]]
        if ispe == 0:
            plt.plot(recall_intpl, precision_intpl, label='eye infomation + textbox number + word number (AUC={})'.format(str(prauc[0])[0:6]),marker="o", markeredgewidth=0, markersize=4)
        elif ispe == 1:
            plt.plot(recall_intpl, precision_intpl, label='textbox number + word number (AUC={})'.format(str(prauc[0])[0:6]),marker="o", markeredgewidth=0, markersize=4)
        elif ispe == 2:
            plt.plot(recall_intpl, precision_intpl, label='minimum word frequency (AUC={})'.format(str(prauc[0])[0:6]),marker="o", markeredgewidth=0, markersize=4)
        elif ispe == 3:
            plt.plot(recall_intpl, precision_intpl, label='predict all label 1 (AUC={})'.format(str(prauc[0])[0:6]),marker="o", markeredgewidth=0, markersize=4)
        # elif ispe == 4:
        #     plt.plot(recall_intpl, precision_intpl, label='predict all label 0 (AUC={})'.format(str(prauc[0])[0:6]),marker="o", markeredgewidth=0, markersize=4)
        ispe +=1
    plt.legend(loc="upper right")
    plt.title('Precision-Recall curve of Understanding Estimation')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    output_pr_file_path = os.path.join(result_dir_path,'_pr_baseline_minfreq_botukamo.png')
    plt.savefig(output_pr_file_path)

    plt.clf()
    plt.close()

    print(output_pr_file_path,' done')

    print(all_pr_auc)

    # header = ['user_id','AUC']
    # with open("../processed/all_10select/merge/all_pr_auc.csv",'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(header)
    #     for d in all_pr_auc:
    #         writer.writerow(d)

# ユーザごとの吹き出しのラベル数を求める
def Count_txbox_number():
    # data_path = "../processed/all_10select/merge/merge.csv"
    data_path = "../dataset/newtxboxdataset.csv"

    # 個別に全員したいとき
    users = []
    for i in range(30):
        n = str(i)
        n = n.zfill(2)
        x = 'P'+n
        users.append(x)

    # users = ['all']

    data = []

    write_data = []

    for u in users:
        if u == 'P00': continue
        if u == 'P22': continue
        if u == 'P25': continue
        if u == 'P29': continue
        print(u)

        tb_label = Read_csv_columns(data_path,'label',user = u)

        s_bool = tb_label == 1
        label1_num = s_bool.sum()
        s_bool = tb_label == 0
        label0_num = s_bool.sum()

        print('label1:',label1_num,'label0:',label0_num)

        label1rate = float(label1_num)/float(label1_num+label0_num)
        print('label1 rate:',label1rate)

        write_data += [[u,label1_num,label0_num,label1rate]]

    header = ['user_id','label1','label0','label1_rate']
    with open('../processed/all_10select/merge/txbox_label1_rate.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for d in write_data:
            writer.writerow(d)

# 2つのファイルを横方向に結合
def Connect_data():
    df1 = pd.read_csv('../processed/all_10select/merge/all_pr_auc.csv')
    df2 = pd.read_csv('../processed/all_10select/merge/txbox_label1_rate.csv')

    df_concat = pd.concat([df2,df1['AUC']],axis=1)

    df_concat.to_csv('../processed/all_10select/merge/all_pr_auc_l1rate.csv',index=False)

# ファイル書き込み
def Write_data():
    header = ['user_id','Neuroticism','Extraversion','Openness','Agreeableness','Conscientiousness','AUC']
    with open("../google/big5_result_label.csv",'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for d in data:
            writer.writerow(d)
    
# Big5関連
# Big5テストの回答から被験者ごとに性格5因子を求める
def Big5(big5_path,save_path):
    # 作成するファイルの準備
    # NEOAC = N:神経症傾向（Neuroticism）E:外向性（Extraversion）O:開放性（Openness）A:協調性（Agreeableness）C:誠実性（Conscientiousness)
    header = ['user_id','Neuroticism','Extraversion','Openness','Agreeableness','Conscientiousness']
    with open(save_path,'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # Big5テストデータの読み込み
    big5_data=[]
    with open(big5_path, encoding='utf-8') as f:
        reader=csv.reader(f)
        header=next(reader)
        for r in reader:
            big5_data.append(r)

    # 各被験者ごとに性格5因子を求める
    # /disk021/share/takahashi/B4exp/google/Big5-50.csvに各項目の特性と反転するかどうか書いている
    # 質問は全部で51問
    # 最後の質問　私は正確かつ正直に答えました　はカウントしない
    # 外向性,協調性,誠実性,神経症的傾向,開放性の順に質問が並んでいる
    # 2,4,6,8,10,12,14,16,18,20,22,24,26,28,29,30,32,34,36,38,39,44,46,49問目は反転

    reverse = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,29,30,32,34,36,38,39,44,46,49]

    for big5_data_raw in big5_data:

        user_id = big5_data_raw[1]

        big5 = [0,0,0,0,0]

        answer = {'あてはまらない' : 1 , 'ややあてはまらない' : 2 , 'どちらともいえない' : 3 , 'ややあてはまる' : 4 , 'あてはまる' : 5}
        for i, d in enumerate(big5_data_raw[2:52]):
            y = answer[d]

            # reverseに入ってたら反転
            if i+1 in reverse:
                y = 6-y

            big5[i%5] += y

        # big5[0]:外向性 big5[1]:協調性 big5[2]:誠実性 big5[3]:神経症的傾向 big5[4]:開放性
        N = big5[3]
        E = big5[0]
        O = big5[4]
        A = big5[1]
        C = big5[2]

        result_data = [user_id,N,E,O,A,C]

        with open(save_path,'a') as f:
            writer = csv.writer(f)
            writer.writerow(result_data)

# 被験者ごとの性格5因子のデータとAUCのデータを結合
def Concat_label(result_path,auc_path):
    # Big5テスト結果の読み込み
    big5_result_data=[]
    with open(result_path,'r') as f:
        reader=csv.reader(f)
        header=next(reader)
        for r in reader:
            big5_result_data.append(r)

    # AUCデータファイルの読み込み
    auc_data=[]
    with open(auc_path,'r') as f:
        reader=csv.reader(f)
        header=next(reader)
        for r in reader:
            auc_data.append(r)

    with_auc_data = []
    # ラベルを配列の末尾に結合
    for brd in big5_result_data:
        user = brd[0]
        for ad in auc_data:
            if user == ad[0]:
                brd.append(ad[1])
                with_auc_data.append(brd)

    print(with_auc_data)

    header = ['user_id','Neuroticism','Extraversion','Openness','Agreeableness','Conscientiousness','AUC']
    with open("../google/big5_result_label_prauc.csv",'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for wad in with_auc_data:
            writer.writerow(wad)

# 性格5因子とAUCのデータから散布図を作成
def Plot_big5_result(big5_result_path):

    df = pd.read_csv(big5_result_path)

    # 性格5因子のすべての組み合わせの散布図作成
    label = ['Neuroticism','Extraversion','Openness','Agreeableness','Conscientiousness']
    label_combi = list(itertools.combinations(label,2))
    # print(label_combi)

    auc = df['AUC'].values
    for lc in label_combi:
        x_label = lc[0]
        y_label = lc[1]
        x = df[x_label].values
        y = df[y_label].values
        save_file_name = x_label + '-' + y_label
        Plot_scatter(x,y,save_file_name,x_label,y_label,auc)

    # PCAで2次元にしたのち散布図作成
    
    X = df[['Neuroticism','Extraversion','Openness','Agreeableness','Conscientiousness']].values

    #pca3d = PCA(n_components=3)
    pca2d = PCA(n_components=2)

    #モデルのパラメータをfitして取得しPCAオブジェクトへ格納
    #pca3d.fit(X)
    pca2d.fit(X)

    #fitで取得したパラメータを参考にXを変換する
    #pca_X3d = pca3d.transform(X)
    pca_X2d = pca2d.transform(X)

    print(pca_X2d)

    save_file_name = 'PCA'

    x = [i[0] for i in pca_X2d]
    y = [i[1] for i in pca_X2d]

    Plot_scatter(x,y,save_file_name,X_label = 'pc1',Y_label = 'pc2',label = auc)

# numpy配列でデータを渡すと散布図作成、labelを渡すとその数値に基づくグラデーション作成
def Plot_scatter(X_data,Y_data,save,X_label = 'X',Y_label = 'Y',label = None):
    plt.scatter(X_data,Y_data,c=label, cmap='Blues')
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.colorbar()
    savefolda = "../google/scatter_prauc"
    # フォルダが存在していなければ作成
    if not os.path.exists(savefolda):
        os.makedirs(savefolda)
    savepath = os.path.join(savefolda,save + '.png')
    plt.savefig(savepath)
    plt.clf()
    plt.close()

# ページ内の吹き出しの数とfixationの数の相関を求めたい，相関係数を求める
def Mk_corr():
    path = '../dataset/dataset_page_fromtxbox.csv'
    df = pd.read_csv(path)
    corr = df.corr()

    savepath = '../dataset/dataset_page_fromtxbox_corr.csv'
    corr.to_csv(savepath)

# 漫画のテキスト情報，具体的にはテキストボックス数，単語数をまとめる
def Mk_manga_info():
    txbox_data_path = '../dataset/converttxbox.csv'
    df_txbox = pd.read_csv(txbox_data_path)
    # 重複があればTrueを返す，uniqueで検索
    # print(df_txbox[df_txbox.duplicated(subset=['work_id','episode_id','page_id'])])
    df_txbox_group_size =  df_txbox.groupby(['work_id','episode_id','page_id']).size().reset_index()
    df_txbox_group_sum =  df_txbox.groupby(['work_id','episode_id','page_id']).sum().reset_index()

    
    print('tb_id sum')
    print(df_txbox_group_size)
    print('wb_num sum')
    # print(df_txbox_group_sum['wb_num'])

    df_manga_info_page = pd.concat([df_txbox_group_size,df_txbox_group_sum['wb_num']],axis=1)
    df_manga_info_page = df_manga_info_page.rename(columns={0:'tb_num'})
    
    df_manga_info_page = df_manga_info_page.astype(str)
    savepath = '../dataset/manga_page_info.csv'
    df_manga_info_page.to_csv(savepath,index=False)
    # print('wb_num mean')
    # print(df_txbox_group_mean['wb_num'])

    # print(df_txbox_group_agg)
    
# 各被験者が何のページをよんだかまとめる
def Mk_read_manga_page():
    readinglog_path =  glob('../rawdata/readingLog/*/*.csv')
    # print(len(readinglog_path))

    data = []
    for path in readinglog_path:
        with open(path,'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for r in reader:
                data.append(r)

    # print(data)
    # print(len(data))

    # li = []
    # sum_df = 0
    # for i,path in enumerate(readinglog_path):
    #     df = pd.read_csv(path,dtype = 'object')
    #     sum_df += len(df)
    #     li.append(df)
    
    # print(sum_df)
    # print(i)
    # # print(li)
    # print(header)
    header = ['user_id', 'work_id', 'episode_id', 'time', 'operation', 'page_id', 'page_path']

    frame = pd.DataFrame(data,columns=header)
    # frame = pd.concat(li,axis=0)

    # print(frame)

    # print(frame)
    
    frame = frame['operation'] == 'backward'
    print(frame.sum())

    # df_user_page =  frame.drop_duplicates(['user_id','work_id','episode_id','page_id'])
    # df_user_page =  frame[frame.duplicated(['user_id','work_id','episode_id','page_id'])]
    # print(df_user_page)
    # print(len(df_user_page))
    # df_user_page = df_user_page[['user_id','work_id','episode_id','page_id']]
    # df_user_page = df_user_page.sort_values('user_id')
    # df_user_page = df_user_page[df_user_page['page_id'] != '99999']
    # df_user_page = df_user_page[df_user_page['page_id'] != '00000']
    # print(len(df_user_page))
    # df_user_page = df_user_page.astype(str)
    # print(len(df_user_page))
    # print(df_user_page)

    # savepath = '../dataset/read_user_page_readback.csv'
    # df_user_page.to_csv(savepath,index=False)
    # for path in readinglog_path:
    #     tmp = path.split('/')
    #     pc_name = tmp[3]

    #     datas = []
    #     with open(path,'r') as f:
    #         reader = csv.reader(f)
    #         header = next(reader)
    #         for r in reader:
    #             datas.append(r)

    #     for data in datas:
    #         user = data[0]
    #         title = data[1]
    #         epi = data[2]
    #         page = data[5]

# ページレベルでの新しいラベル作成
def Mk_page_label():
    txbox_label_path = "../dataset/truelabel.csv"
    df_txbox_label = pd.read_csv(txbox_label_path, dtype="object")

    # df_page_group = df_txbox_label.drop_duplicates(['user_id','work_id','episode_id','page_id'])
    # df_page_group = df_page_group[['user_id','work_id','episode_id','page_id']]
    # print(type(df_page_group))
    # header = ['user_id', 'work_id', 'episode_id', 'page_id', 'label']
    df_page_group_size =  df_txbox_label.groupby(['user_id','work_id','episode_id','page_id']).size().reset_index()
    df_page_group_size = df_page_group_size.rename(columns={0:'label'})
    df_page_group_size = df_page_group_size.astype(str)
    savepath = "../dataset/truelabel_page.csv"
    df_page_group_size.to_csv(savepath,index=False)

    # print(type(df_page_group_size))

    # label = pd.DataFrame(df_page_group_size,columns=['label'])
    
    # print(type(label))

    # 

    

    # print(type(df_page_group_size))

    

    # print(df_page_group)

    # df = pd.concat([df_page_group,label],axis=1)

    # print(df)
    # df.to_csv(savepath,index=False)

# データセットの基本情報をまとめる，ユーザー名，タイトル，エピソード，ページ，ラベル(わからない吹き出しの合計数)，テキストボックス数，単語数
def Mk_basic_dataset():
    # 被験者が読んだすべてのページを記録
    readpage_path = "../dataset/read_user_page.csv"
    # ページごとのラベル
    pagelabel_path = "../dataset/truelabel_page.csv"
    # 漫画ごとのテキストボックス数，単語数が書いてある
    info_path = "../dataset/manga_page_info2.csv"

    readpage_df = pd.read_csv(readpage_path, dtype="object")
    pagelabel_df = pd.read_csv(pagelabel_path, dtype="object")
    info_df = pd.read_csv(info_path, dtype="object")

    df = pd.merge(readpage_df,pagelabel_df,how='left')
    df = pd.merge(df,info_df,how='left')

    # テキストボックスがないページ，もしくはミスによる重複？でデータがないページが存在したdataset_base_original.csvに保存
    # 理解度推定においてそのようなページは必要なため削除
    df = df.dropna(subset = ['tb_num','wb_num'])
    df = df.fillna({'label':0})
    savepath = "../dataset/dataset_base2.csv"
    df.to_csv(savepath,index=False)

# 被験者ごとのfixationの平均の平均を求めてみる
def Avr_fix_num():
    target_columns = 'fix_dur_sum_sd'
    target_path = "../dataset/dataset_page_fromtxbox.csv"
    df = pd.read_csv(target_path)

    df = df[['user_id',target_columns]]

    df = df.groupby(['user_id']).mean().reset_index()

    df = df.rename(columns={0:target_columns})

    save_path = os.path.join("../dataset",target_columns+".csv")
    df.to_csv(save_path,index=False)

# baseline作成，具体的にはわからない吹き出しの割合を求めてソートした結果とする
def Mk_baseline():
    # これすると0か0以外でラベル分けされてるから意味ない笑
    # dataset_path = "../dataset/dataset_base.csv"
    # dataset_df = pd.read_csv(dataset_path)

    # dataset_df['label1_rate'] = dataset_df['label'] / dataset_df['tb_num']

    # dataset_df.to_csv("../dataset/label1rate.csv",index=False)

    # baselineの作成，考えているのは文書情報のみから得られる特徴量を用いた推定，テキストボックス数，単語数，(単語出現頻度は時間あれば)
    dataset_path = "../dataset/dataset_page_fromtxbox.csv"
    dataset_df = pd.read_csv(dataset_path)

    baseline_df = dataset_df[['user_id','work_id','episode_id','page_id','label','tb_num','wb_num_sum','wb_num_max','wb_num_min','wb_num_avr','wb_num_var','wb_num_sd']]
    save_path = "../dataset/dataset_baseline.csv"
    baseline_df.to_csv(save_path,index=False)
    # テキストボックスだけでいけるんだが，ラベルの取り方間違えてる説

def Mk_newtxboxdataset_page():
    newtxboxdataset_path = "../dataset/newtxboxdataset.csv"
    newtxboxdataset_df = pd.read_csv(newtxboxdataset_path)

    df_page_group_size =  newtxboxdataset_df.groupby(['user_id','work_id','episode_id','page_id']).sum().reset_index()

    savepath = "../dataset/newtxboxdataset_page.csv"
    df_page_group_size.to_csv(savepath,index=False)

def Mk_new_complete_dataset():
    p = "../dataset/dataset_base.csv"
    q = "../dataset/newtxboxdataset_page.csv"

    p_df = pd.read_csv(p)
    q_df = pd.read_csv(q)

    df = pd.merge(p_df,q_df[6:],how='left')

    savepath = "../dataset/new_complete_dataset.csv"
    df.to_csv(savepath,index=False)

def Free():
    # p = "../fsUserExtractor/fsUser_15_50_80.csv"
    # p_df = pd.read_csv(p, dtype="object")
    # df_page=  p_df.groupby(['user_id','work_id','episode_id','page_id']).sum().reset_index()
    # df_page = df_page[['user_id','work_id','episode_id','page_id','fixation_duration','saccade_length']]
    # df_page = df_page.astype(str)
    # savepath = "../koredeeeyan.csv"
    # df_page.to_csv(savepath,index=False)

    # p = "../dataset/dataset_base.csv"
    # q = "../koredeeeyan2.csv"
    # p_df = pd.read_csv(p, dtype="object")
    # q_df = pd.read_csv(q, dtype="object")
    # df = pd.merge(p_df,q_df[4:],how='left')
    # df = df.astype(str)
    # savepath = "../haiowari.csv"
    # df.to_csv(savepath,index=False)

    # path = "../koredeeeyan.csv"
    # data = []
    # with open(path,'r') as f:
    #     reader = csv.reader(f)
    #     header = next(reader)
    #     for r in reader:
    #         data.append(r)
    # path = "../koredeeeyan2.csv"
    # with open(path,'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(header)

    # for d in data:
    #     user = d[0]
    #     title = d[1]
    #     epi = d[2].zfill(3)
    #     page = d[3].zfill(5)
    #     info = [user,title,epi,page,d[4],d[5]]
    #     with open(path,'a') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(info)

    # p = "../compare/mergesvmresult10bss.csv"
    # p_df = pd.read_csv(p)
    # p_df = p_df.groupby(['user_id','work_id','episode_id','page_id']).sum().reset_index()
    # p_df = p_df[['user_id','work_id','episode_id','page_id','y_true','y_pred']]
    # savepath = "../compare/daiku_page.csv"
    # p_df.to_csv(savepath,index=False)

    path = "../compare/daiku_page.csv"
    data = []
    with open(path,'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            data.append(r)

    path = "../compare/daiku_page2.csv"
    with open(path,'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for d in data:
        user = d[0]
        title = d[1]
        epi = d[2].zfill(3)
        page = d[3].zfill(5)
        info = [user,title,epi,page,d[4],d[5]]
        with open(path,'a') as f:
            writer = csv.writer(f)
            writer.writerow(info)

def Mk_minfreq_page():
    p = "../dataset/dataset2.csv"
    p_df = pd.read_csv(p, dtype="object")
    df_page_group_size =  p_df.groupby(['user_id','work_id','episode_id','page_id']).min().reset_index()
    df_page_group_size = df_page_group_size[['user_id','work_id','episode_id','page_id','min_freq']]
    df_page_group_size = df_page_group_size.astype(str)
    q = "../dataset/dataset_base.csv"
    q_df = pd.read_csv(q, dtype="object")
    df = pd.merge(q_df,df_page_group_size,how='left')
    df.to_csv("../minfreq_page.csv",index=False)
        



def main(): 
    # Count_txbox_number()
    # Mk_pr()
    # Connect_data()
    # Mk_corr()
    # Mk_manga_info()
    # Mk_read_manga_page()
    # Mk_page_label()
    Mk_basic_dataset()
    # Avr_fix_num()
    # Mk_baseline()
    # Mk_newtxboxdataset_page()
    # Mk_new_complete_dataset()
    # Free()
    # path = "../processed_haiowaei/all_all_select/merge/merge.csv"
    # pred = Read_csv_columns(path,'y_pred')
    # true = Read_csv_columns(path,'y_true')
    # print(Calc_accuracy(true,pred))
    # print(Label1_num(true))
    # print(confusion_matrix(true, pred))
    # Mk_minfreq_page()

    # ../compare/daiku_page2.csv
    # path = "../processed_haiowaei/all_all_select/merge/merge.csv"
    # pred = Read_csv_columns(path,'y_pred')
    # true = Read_csv_columns(path,'y_true')

    # print(pred)
    # print(true)

    # pred[pred!=0]=1
    # true[true!=0]=1
    
    # print(pred)
    # print(true)

    # print(Calc_accuracy(true,pred))
    # print(Label1_num(true))
    # print(confusion_matrix(true, pred))

    # p = "../processed_haiowaei/all_all_select/merge/merge.csv"
    # data = []
    # with open(p,'r') as f:
    #     reader = csv.reader(f)
    #     header = next(reader)
    #     for r in reader:
    #         data.append(r)
    
    # p = "../allpred-1.csv"
    # with open(p,'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(header)
    #     for d in data:
    #         d[7]=-1.0
    #         writer.writerow(d)




    

if __name__ == "__main__":
    main()