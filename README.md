# word2vec+cnn_shortAd_Classification
## 微信广告正负样本分类
参考https://github.com/clayandgithub/zh_cnn_text_classify.git
## Dependencies
tensorflow (1.6 run success)
未做分词，直接单个词处理

正样本：

* xx,xxx,xxxxxxxxx

负样本：

* 微信公众号 AppSo，回复「钱包」看看微信钱包这 6 个秘密使用技巧
 
* 微信号：wszs1981
 
* 长按二维码关注
 

训练数据集：data/ad_5000.txt 、 data/not_ad_4000.txt

    python train.py

测试数据集：data/ad_5000.txt ad_5000_label、data/not_ad_4000.txt not_ad_4000_label 、 data/test.txt  test_label
查看运行结果：

    python eval.py

## result

    #data/prediction_ad_5000.txt
    Total number of test examples: 5000
    Accuracy: 0.9944
   
    #data/prediction_not_ad_4000.txt
    Total number of test examples: 4000
    Accuracy: 0.99475

    #对于某些样本泛化能力不强 data/prediction_test.txt
    市民首次使用二维码乘车，可通过微信小程序搜索“腾讯乘车码”，此后乘车码会自动进入微信卡包，再次使用时在“我-卡包-会员卡”可以轻松找到，乘坐公交时靠近扫码机，刷码成功后即可乘车。
	b'0.0'
    首次使用二维码乘车。
	b'0.0'
    五要谨慎选择支付平台，网购尽量选择信誉度高的卖家购买，并通过正规的第三方支付渠道进行支付，谨慎使用“二维码”支付，防止不法分子盗取个人账户等信息。
	b'0.0'
    


    

