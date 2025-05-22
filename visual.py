from nlgeval import compute_metrics

metrics_dict = compute_metrics(hypothesis='/workspace/nlp-cgi/outputs/x_NLMCXR_ClsGenInt_DenseNet121_MaxView2_NumLabel114_Retrieval_2History_Hyp.txt',
                               references=['/workspace/nlp-cgi/outputs/x_NLMCXR_ClsGenInt_DenseNet121_MaxView2_NumLabel114_Retrieval_2History_Ref.txt'])
# metrics_dict = compute_metrics(hypothesis='/workspace/cmn/results/iu_xray/x_test_Hyp.txt',
#                                references=['/workspace/cmn/results/iu_xray/x_test_Ref.txt'])

print(metrics_dict)