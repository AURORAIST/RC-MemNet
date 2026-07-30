# Authentic Experiment Data Audit & Real CSV Summary Report

## 1. num_prompt_sweep_8targets_summary.csv Content

**Path**: `/home/ustc1958/lxy/graph/tone/complete_package0712/output/0614/num_prompt_sweep_8targets_dryrun/num_prompt_sweep_8targets_summary.csv`

```csv
   accuracy      area  best_step  best_val_macro_f1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             command  curve_rows  final_step  gate_rows  global_accuracy  global_macro_f1  global_weighted_f1 holdout  macro_f1  memory_rows  micro_f1  num_prompts  pcmr_rows  prediction_rows region_column                                                                      result_file  returncode  routing_rows                                                              run_dir   status stopped_early  weighted_f1
0       NaN  Qingyang        NaN                NaN  /home/ustc1958/miniconda3/envs/graph/bin/python -u /home/ustc1958/lxy/graph/tone/complete_package0614/tools/experiments/paper_dual_memory.py --csv data/manifests/wu_low_resource_vowel_dataset.fixed_paths.csv --region-column region --holdout-region 04青阳 --device cuda --eval-support-shots 4 --eval-split-runs 100 --eval-split-mode global_support --support-shots 2 --max-steps 3000 --early-stop-patience 200 --early-stop-min-delta 0.0001 --early-stop-min-steps 300 --lr 0.0003 --seed 0 --lambda-global 0.5 --hidden-dim 256 --score-dim 128 --prompt-dim 128 --num-prompts 2 --layers 3 --heads 4 --ffn-dim 768 --dropout 0.1 --temperature 0.2 --episode-length 32 --episode-batch-size 2 --eval-ensemble-runs 1 --log-every 50 --eval-every 50 --save-checkpoint --output output/0614/num_prompt_sweep_8targets_dryrun/num_prompts_2/Qingyang/result.json --curve-output output/0614/num_prompt_sweep_8targets_dryrun/num_prompts_2/Qingyang/curve.csv --debug-output output/0614/num_prompt_sweep_8targets_dryrun/num_prompts_2/Qingyang/debug.json --audit-output output/0614/num_prompt_sweep_8targets_dryrun/num_prompts_2/Qingyang/audit.json --predictions-output output/0614/num_prompt_sweep_8targets_dryrun/num_prompts_2/Qingyang/predictions.csv --mechanism-output-dir output/0614/num_prompt_sweep_8targets_dryrun/num_prompts_2/Qingyang/csv --mechanism-max-episodes 0 --mechanism-k-values 1 3 5 10         NaN         NaN        NaN              NaN              NaN                 NaN    04青阳       NaN          NaN       NaN            2        NaN              NaN        region  output/0614/num_prompt_sweep_8targets_dryrun/num_prompts_2/Qingyang/result.json           0           NaN  output/0614/num_prompt_sweep_8targets_dryrun/num_prompts_2/Qingyang  dry_run           NaN          NaN
1  0.623508   Chizhou     1100.0           0.498288        /home/ustc1958/miniconda3/envs/graph/bin/python -u /home/ustc1958/lxy/graph/tone/complete_package0614/tools/experiments/paper_dual_memory.py --csv data/manifests/wu_low_resource_vowel_dataset.fixed_paths.csv --region-column region --holdout-region 03池州 --device cuda --eval-support-shots 4 --eval-split-runs 100 --eval-split-mode global_support --support-shots 2 --max-steps 3000 --early-stop-patience 200 --early-stop-min-delta 0.0001 --early-stop-min-steps 300 --lr 0.0003 --seed 0 --lambda-global 0.5 --hidden-dim 256 --score-dim 128 --prompt-dim 128 --num-prompts 2 --layers 3 --heads 4 --ffn-dim 768 --dropout 0.1 --temperature 0.2 --episode-length 32 --episode-batch-size 2 --eval-ensemble-runs 1 --log-every 50 --eval-every 50 --save-checkpoint --output output/0614/num_prompt_sweep_8targets_dryrun/num_prompts_2/Chizhou/result.json --curve-output output/0614/num_prompt_sweep_8targets_dryrun/num_prompts_2/Chizhou/curve.csv --debug-output output/0614/num_prompt_sweep_8targets_dryrun/num_prompts_2/Chizhou/debug.json --audit-output output/0614/num_prompt_sweep_8targets_dryrun/num_prompts_2/Chizhou/audit.json --predictions-output output/0614/num_prompt_sweep_8targets_dryrun/num_prompts_2/Chizhou/predictions.csv --mechanism-output-dir output/0614/num_prompt_sweep_8targets_dryrun/num_prompts_2/Chizhou/csv --mechanism-max-episodes 0 --mechanism-k-values 1 3 5 10      1300.0      1300.0       36.0         0.650348         0.510689            0.626709    03池州  0.552506        108.0  0.623508            2   626292.0          21645.0        region   output/0614/num_prompt_sweep_8targets_dryrun/num_prompts_2/Chizhou/result.json           0       43290.0   output/0614/num_prompt_sweep_8targets_dryrun/num_prompts_2/Chizhou   copied          True     0.630229
```

### 8-Region Prompt Sweep (L = 2, 4, 6, 8, 10) Results Matrix
**Source Directory**: `/home/ustc1958/lxy/graph/tone/complete_package0712/output/0614/num_prompt_sweep_8targets_20260629`

```text
 L Qingyang Tongling Jingxian Nanling Ningguo Lishui Chizhou Huangshan 8-Region Avg
 2   60.05%   44.58%   45.28%  59.36%  51.17% 56.89%  62.35%    52.05%       53.97%
 4   55.81%   52.11%   47.23%  61.48%  60.79% 50.34%  64.15%    48.47%       55.05%
 6   49.59%   39.39%   46.28%  53.11%  44.71% 49.40%  58.29%    49.40%       48.77%
 8   60.56%   49.66%   21.97%  53.07%  54.57% 51.47%  62.77%    55.04%       51.14%
10   56.73%   50.87%   38.55%  57.03%  50.36% 55.86%  60.77%    47.94%       52.26%
```

## 2. 8region_kshot_model_mean_area_std.csv Content

**Path**: `/home/ustc1958/lxy/graph/tone/model/8region_kshot_figures/8region_kshot_model_mean_area_std.csv`

```csv
                    model   k  accuracy_mean  accuracy_area_std       metric  macro_f1_mean  macro_f1_area_std  weighted_f1_mean  weighted_f1_area_std
0    PC-DLCMNet-prototype   1       0.485351           0.028235     accuracy            NaN                NaN               NaN                   NaN
1    PC-DLCMNet-prototype   2       0.525806           0.024758     accuracy            NaN                NaN               NaN                   NaN
2    PC-DLCMNet-prototype   3       0.544182           0.025738     accuracy            NaN                NaN               NaN                   NaN
3    PC-DLCMNet-prototype   4       0.551413           0.026142     accuracy            NaN                NaN               NaN                   NaN
4    PC-DLCMNet-prototype   5       0.557518           0.025040     accuracy            NaN                NaN               NaN                   NaN
5    PC-DLCMNet-prototype   6       0.561294           0.028052     accuracy            NaN                NaN               NaN                   NaN
6    PC-DLCMNet-prototype   7       0.560129           0.025979     accuracy            NaN                NaN               NaN                   NaN
7    PC-DLCMNet-prototype   8       0.560395           0.025788     accuracy            NaN                NaN               NaN                   NaN
8    PC-DLCMNet-prototype   9       0.562823           0.027786     accuracy            NaN                NaN               NaN                   NaN
9    PC-DLCMNet-prototype  10       0.562320           0.028231     accuracy            NaN                NaN               NaN                   NaN
10            mHuBERT-147   1       0.215002           0.022740     accuracy            NaN                NaN               NaN                   NaN
11            mHuBERT-147   2       0.256714           0.034015     accuracy            NaN                NaN               NaN                   NaN
12            mHuBERT-147   3       0.282533           0.039242     accuracy            NaN                NaN               NaN                   NaN
13            mHuBERT-147   4       0.300763           0.044835     accuracy            NaN                NaN               NaN                   NaN
14            mHuBERT-147   5       0.315774           0.047609     accuracy            NaN                NaN               NaN                   NaN
15            mHuBERT-147   6       0.328362           0.049327     accuracy            NaN                NaN               NaN                   NaN
16            mHuBERT-147   7       0.336950           0.050477     accuracy            NaN                NaN               NaN                   NaN
17            mHuBERT-147   8       0.344356           0.051648     accuracy            NaN                NaN               NaN                   NaN
18            mHuBERT-147   9       0.350320           0.051864     accuracy            NaN                NaN               NaN                   NaN
19            mHuBERT-147  10       0.355683           0.052872     accuracy            NaN                NaN               NaN                   NaN
20              MR-HuBERT   1       0.221798           0.027717     accuracy            NaN                NaN               NaN                   NaN
21              MR-HuBERT   2       0.259892           0.039376     accuracy            NaN                NaN               NaN                   NaN
22              MR-HuBERT   3       0.283105           0.047189     accuracy            NaN                NaN               NaN                   NaN
23              MR-HuBERT   4       0.296880           0.052441     accuracy            NaN                NaN               NaN                   NaN
24              MR-HuBERT   5       0.308560           0.055756     accuracy            NaN                NaN               NaN                   NaN
25              MR-HuBERT   6       0.318207           0.058549     accuracy            NaN                NaN               NaN                   NaN
26              MR-HuBERT   7       0.324174           0.060104     accuracy            NaN                NaN               NaN                   NaN
27              MR-HuBERT   8       0.328821           0.062606     accuracy            NaN                NaN               NaN                   NaN
28              MR-HuBERT   9       0.333564           0.063216     accuracy            NaN                NaN               NaN                   NaN
29              MR-HuBERT  10       0.336948           0.063584     accuracy            NaN                NaN               NaN                   NaN
30              MS-HuBERT   1       0.219758           0.027919     accuracy            NaN                NaN               NaN                   NaN
31              MS-HuBERT   2       0.255407           0.037294     accuracy            NaN                NaN               NaN                   NaN
32              MS-HuBERT   3       0.274959           0.042476     accuracy            NaN                NaN               NaN                   NaN
33              MS-HuBERT   4       0.289689           0.045865     accuracy            NaN                NaN               NaN                   NaN
34              MS-HuBERT   5       0.299934           0.049130     accuracy            NaN                NaN               NaN                   NaN
35              MS-HuBERT   6       0.307517           0.051127     accuracy            NaN                NaN               NaN                   NaN
36              MS-HuBERT   7       0.312703           0.052675     accuracy            NaN                NaN               NaN                   NaN
37              MS-HuBERT   8       0.316706           0.054934     accuracy            NaN                NaN               NaN                   NaN
38              MS-HuBERT   9       0.319775           0.055122     accuracy            NaN                NaN               NaN                   NaN
39              MS-HuBERT  10       0.323137           0.055451     accuracy            NaN                NaN               NaN                   NaN
40          SALMONN-proxy   1       0.158158           0.009133     accuracy            NaN                NaN               NaN                   NaN
41          SALMONN-proxy   2       0.177591           0.015380     accuracy            NaN                NaN               NaN                   NaN
42          SALMONN-proxy   3       0.189859           0.017798     accuracy            NaN                NaN               NaN                   NaN
43          SALMONN-proxy   4       0.198322           0.019189     accuracy            NaN                NaN               NaN                   NaN
44          SALMONN-proxy   5       0.205002           0.022149     accuracy            NaN                NaN               NaN                   NaN
45          SALMONN-proxy   6       0.209507           0.023703     accuracy            NaN                NaN               NaN                   NaN
46          SALMONN-proxy   7       0.214035           0.024786     accuracy            NaN                NaN               NaN                   NaN
47          SALMONN-proxy   8       0.218605           0.026874     accuracy            NaN                NaN               NaN                   NaN
48          SALMONN-proxy   9       0.221514           0.027359     accuracy            NaN                NaN               NaN                   NaN
49          SALMONN-proxy  10       0.225035           0.027610     accuracy            NaN                NaN               NaN                   NaN
50   PC-DLCMNet-prototype   1            NaN                NaN     macro_f1       0.473519           0.023026               NaN                   NaN
51   PC-DLCMNet-prototype   2            NaN                NaN     macro_f1       0.510409           0.025836               NaN                   NaN
52   PC-DLCMNet-prototype   3            NaN                NaN     macro_f1       0.526687           0.026016               NaN                   NaN
53   PC-DLCMNet-prototype   4            NaN                NaN     macro_f1       0.532128           0.025393               NaN                   NaN
54   PC-DLCMNet-prototype   5            NaN                NaN     macro_f1       0.537845           0.019984               NaN                   NaN
55   PC-DLCMNet-prototype   6            NaN                NaN     macro_f1       0.540943           0.025139               NaN                   NaN
56   PC-DLCMNet-prototype   7            NaN                NaN     macro_f1       0.542140           0.028325               NaN                   NaN
57   PC-DLCMNet-prototype   8            NaN                NaN     macro_f1       0.541706           0.026928               NaN                   NaN
58   PC-DLCMNet-prototype   9            NaN                NaN     macro_f1       0.542231           0.025948               NaN                   NaN
59   PC-DLCMNet-prototype  10            NaN                NaN     macro_f1       0.541529           0.025012               NaN                   NaN
60            mHuBERT-147   1            NaN                NaN     macro_f1       0.193869           0.022855               NaN                   NaN
61            mHuBERT-147   2            NaN                NaN     macro_f1       0.236045           0.032892               NaN                   NaN
62            mHuBERT-147   3            NaN                NaN     macro_f1       0.260838           0.036771               NaN                   NaN
63            mHuBERT-147   4            NaN                NaN     macro_f1       0.277462           0.040644               NaN                   NaN
64            mHuBERT-147   5            NaN                NaN     macro_f1       0.290941           0.042326               NaN                   NaN
65            mHuBERT-147   6            NaN                NaN     macro_f1       0.301855           0.043981               NaN                   NaN
66            mHuBERT-147   7            NaN                NaN     macro_f1       0.309066           0.044569               NaN                   NaN
67            mHuBERT-147   8            NaN                NaN     macro_f1       0.315205           0.045550               NaN                   NaN
68            mHuBERT-147   9            NaN                NaN     macro_f1       0.319705           0.045434               NaN                   NaN
69            mHuBERT-147  10            NaN                NaN     macro_f1       0.323765           0.045746               NaN                   NaN
70              MR-HuBERT   1            NaN                NaN     macro_f1       0.199495           0.027192               NaN                   NaN
71              MR-HuBERT   2            NaN                NaN     macro_f1       0.237761           0.036592               NaN                   NaN
72              MR-HuBERT   3            NaN                NaN     macro_f1       0.259693           0.042033               NaN                   NaN
73              MR-HuBERT   4            NaN                NaN     macro_f1       0.272593           0.045425               NaN                   NaN
74              MR-HuBERT   5            NaN                NaN     macro_f1       0.283052           0.048041               NaN                   NaN
75              MR-HuBERT   6            NaN                NaN     macro_f1       0.291441           0.050412               NaN                   NaN
76              MR-HuBERT   7            NaN                NaN     macro_f1       0.296264           0.051536               NaN                   NaN
77              MR-HuBERT   8            NaN                NaN     macro_f1       0.299914           0.053000               NaN                   NaN
78              MR-HuBERT   9            NaN                NaN     macro_f1       0.303424           0.053375               NaN                   NaN
79              MR-HuBERT  10            NaN                NaN     macro_f1       0.305921           0.053214               NaN                   NaN
80              MS-HuBERT   1            NaN                NaN     macro_f1       0.197907           0.026147               NaN                   NaN
81              MS-HuBERT   2            NaN                NaN     macro_f1       0.233973           0.034444               NaN                   NaN
82              MS-HuBERT   3            NaN                NaN     macro_f1       0.252548           0.037847               NaN                   NaN
83              MS-HuBERT   4            NaN                NaN     macro_f1       0.265373           0.039477               NaN                   NaN
84              MS-HuBERT   5            NaN                NaN     macro_f1       0.274063           0.041318               NaN                   NaN
85              MS-HuBERT   6            NaN                NaN     macro_f1       0.280845           0.042566               NaN                   NaN
86              MS-HuBERT   7            NaN                NaN     macro_f1       0.284932           0.043315               NaN                   NaN
87              MS-HuBERT   8            NaN                NaN     macro_f1       0.287853           0.044628               NaN                   NaN
88              MS-HuBERT   9            NaN                NaN     macro_f1       0.289688           0.044435               NaN                   NaN
89              MS-HuBERT  10            NaN                NaN     macro_f1       0.292181           0.044366               NaN                   NaN
90          SALMONN-proxy   1            NaN                NaN     macro_f1       0.135898           0.010067               NaN                   NaN
91          SALMONN-proxy   2            NaN                NaN     macro_f1       0.156961           0.014658               NaN                   NaN
92          SALMONN-proxy   3            NaN                NaN     macro_f1       0.169490           0.017225               NaN                   NaN
93          SALMONN-proxy   4            NaN                NaN     macro_f1       0.177495           0.018191               NaN                   NaN
94          SALMONN-proxy   5            NaN                NaN     macro_f1       0.182856           0.020604               NaN                   NaN
95          SALMONN-proxy   6            NaN                NaN     macro_f1       0.187109           0.021616               NaN                   NaN
96          SALMONN-proxy   7            NaN                NaN     macro_f1       0.191165           0.022225               NaN                   NaN
97          SALMONN-proxy   8            NaN                NaN     macro_f1       0.194618           0.024041               NaN                   NaN
98          SALMONN-proxy   9            NaN                NaN     macro_f1       0.197271           0.024562               NaN                   NaN
99          SALMONN-proxy  10            NaN                NaN     macro_f1       0.199751           0.024974               NaN                   NaN
100  PC-DLCMNet-prototype   1            NaN                NaN  weighted_f1            NaN                NaN          0.478703              0.028785
101  PC-DLCMNet-prototype   2            NaN                NaN  weighted_f1            NaN                NaN          0.518463              0.026323
102  PC-DLCMNet-prototype   3            NaN                NaN  weighted_f1            NaN                NaN          0.536510              0.027456
103  PC-DLCMNet-prototype   4            NaN                NaN  weighted_f1            NaN                NaN          0.543589              0.027410
104  PC-DLCMNet-prototype   5            NaN                NaN  weighted_f1            NaN                NaN          0.549146              0.025703
105  PC-DLCMNet-prototype   6            NaN                NaN  weighted_f1            NaN                NaN          0.553077              0.029128
106  PC-DLCMNet-prototype   7            NaN                NaN  weighted_f1            NaN                NaN          0.552169              0.027533
107  PC-DLCMNet-prototype   8            NaN                NaN  weighted_f1            NaN                NaN          0.552371              0.027325
108  PC-DLCMNet-prototype   9            NaN                NaN  weighted_f1            NaN                NaN          0.554356              0.028803
109  PC-DLCMNet-prototype  10            NaN                NaN  weighted_f1            NaN                NaN          0.553856              0.029039
110           mHuBERT-147   1            NaN                NaN  weighted_f1            NaN                NaN          0.223476              0.024254
111           mHuBERT-147   2            NaN                NaN  weighted_f1            NaN                NaN          0.269245              0.035303
112           mHuBERT-147   3            NaN                NaN  weighted_f1            NaN                NaN          0.297294              0.041181
113           mHuBERT-147   4            NaN                NaN  weighted_f1            NaN                NaN          0.316670              0.046861
114           mHuBERT-147   5            NaN                NaN  weighted_f1            NaN                NaN          0.332358              0.049350
115           mHuBERT-147   6            NaN                NaN  weighted_f1            NaN                NaN          0.345561              0.051362
116           mHuBERT-147   7            NaN                NaN  weighted_f1            NaN                NaN          0.354565              0.052154
117           mHuBERT-147   8            NaN                NaN  weighted_f1            NaN                NaN          0.362595              0.053312
118           mHuBERT-147   9            NaN                NaN  weighted_f1            NaN                NaN          0.369040              0.053369
119           mHuBERT-147  10            NaN                NaN  weighted_f1            NaN                NaN          0.374678              0.054613
120             MR-HuBERT   1            NaN                NaN  weighted_f1            NaN                NaN          0.227057              0.029604
121             MR-HuBERT   2            NaN                NaN  weighted_f1            NaN                NaN          0.268953              0.041658
122             MR-HuBERT   3            NaN                NaN  weighted_f1            NaN                NaN          0.294308              0.050375
123             MR-HuBERT   4            NaN                NaN  weighted_f1            NaN                NaN          0.308492              0.055742
124             MR-HuBERT   5            NaN                NaN  weighted_f1            NaN                NaN          0.320678              0.059284
125             MR-HuBERT   6            NaN                NaN  weighted_f1            NaN                NaN          0.330857              0.062438
126             MR-HuBERT   7            NaN                NaN  weighted_f1            NaN                NaN          0.337004              0.063664
127             MR-HuBERT   8            NaN                NaN  weighted_f1            NaN                NaN          0.341780              0.066055
128             MR-HuBERT   9            NaN                NaN  weighted_f1            NaN                NaN          0.346943              0.066528
129             MR-HuBERT  10            NaN                NaN  weighted_f1            NaN                NaN          0.350713              0.066943
130             MS-HuBERT   1            NaN                NaN  weighted_f1            NaN                NaN          0.222995              0.027256
131             MS-HuBERT   2            NaN                NaN  weighted_f1            NaN                NaN          0.260444              0.037051
132             MS-HuBERT   3            NaN                NaN  weighted_f1            NaN                NaN          0.280734              0.043189
133             MS-HuBERT   4            NaN                NaN  weighted_f1            NaN                NaN          0.295838              0.047062
134             MS-HuBERT   5            NaN                NaN  weighted_f1            NaN                NaN          0.306204              0.051051
135             MS-HuBERT   6            NaN                NaN  weighted_f1            NaN                NaN          0.314165              0.053049
136             MS-HuBERT   7            NaN                NaN  weighted_f1            NaN                NaN          0.319587              0.054540
137             MS-HuBERT   8            NaN                NaN  weighted_f1            NaN                NaN          0.323898              0.056596
138             MS-HuBERT   9            NaN                NaN  weighted_f1            NaN                NaN          0.326726              0.056567
139             MS-HuBERT  10            NaN                NaN  weighted_f1            NaN                NaN          0.330557              0.056777
140         SALMONN-proxy   1            NaN                NaN  weighted_f1            NaN                NaN          0.165008              0.010068
141         SALMONN-proxy   2            NaN                NaN  weighted_f1            NaN                NaN          0.188337              0.015796
142         SALMONN-proxy   3            NaN                NaN  weighted_f1            NaN                NaN          0.202402              0.019491
143         SALMONN-proxy   4            NaN                NaN  weighted_f1            NaN                NaN          0.211395              0.020731
144         SALMONN-proxy   5            NaN                NaN  weighted_f1            NaN                NaN          0.218331              0.024394
145         SALMONN-proxy   6            NaN                NaN  weighted_f1            NaN                NaN          0.223416              0.025866
146         SALMONN-proxy   7            NaN                NaN  weighted_f1            NaN                NaN          0.228175              0.027083
147         SALMONN-proxy   8            NaN                NaN  weighted_f1            NaN                NaN          0.232963              0.029549
148         SALMONN-proxy   9            NaN                NaN  weighted_f1            NaN                NaN          0.236041              0.030757
149         SALMONN-proxy  10            NaN                NaN  weighted_f1            NaN                NaN          0.240252              0.030989
```

## 3. Search for Memory Items (M) and Write Top-K (K) Sweeps

Found run configurations in output/0722:
```text
                                                                   path                variant num_slots (M) write_top_k (K) accuracy
                                     smoke_cuda_safe_222927/result.json              rc_memnet          None            None   42.22%
        ssl_finetune_whisper03/results/whisper-base/Chizhou/result.json                   None          None            None   50.65%
               ablation_prompt_20260723_001604/11_Huangshan/result.json      no_prompt_routing          None            None   59.35%
               ablation_prompt_20260723_001604/07_Xuancheng/result.json      no_prompt_routing          None            None   68.40%
                 ablation_prompt_20260723_001604/03_Chizhou/result.json      no_prompt_routing          None            None   71.92%
                 ablation_prompt_20260723_001604/12_Ningguo/result.json      no_prompt_routing          None            None   62.91%
                  ablation_prompt_20260723_001604/01_Dangtu/result.json      no_prompt_routing          None            None   54.18%
                  ablation_prompt_20260723_001604/14_Lishui/result.json      no_prompt_routing          None            None   61.38%
                 ablation_prompt_20260723_001604/13_Gaochun/result.json      no_prompt_routing          None            None   52.10%
                  ablation_prompt_20260723_001604/05_Suncun/result.json      no_prompt_routing          None            None   73.21%
                    ablation_prompt_20260723_001604/02_Wuhu/result.json      no_prompt_routing          None            None   69.09%
                ablation_prompt_20260723_001604/08_Jingxian/result.json      no_prompt_routing          None            None   53.92%
                ablation_prompt_20260723_001604/09_Fanchang/result.json      no_prompt_routing          None            None   69.96%
                ablation_prompt_20260723_001604/06_Tongling/result.json      no_prompt_routing          None            None   67.82%
                 ablation_prompt_20260723_001604/10_Nanling/result.json      no_prompt_routing          None            None   67.25%
                ablation_prompt_20260723_001604/04_Qingyang/result.json      no_prompt_routing          None            None   72.36%
                  ablation_fix_03_chizhou/no_memory_adapter/result.json      no_memory_adapter          None            None   70.37%
                ablation_fix_03_chizhou/no_prompt_no_memory/result.json    no_prompt_no_memory          None            None   69.95%
                  ablation_fix_03_chizhou/no_context_prompt/result.json      no_context_prompt          None            None   70.46%
ablation_memory_count_based_v2_20260723_012548/11_Huangshan/result.json count_based_adaptation          None            None   61.85%
ablation_memory_count_based_v2_20260723_012548/07_Xuancheng/result.json count_based_adaptation          None            None   67.83%
  ablation_memory_count_based_v2_20260723_012548/03_Chizhou/result.json count_based_adaptation          None            None   72.12%
  ablation_memory_count_based_v2_20260723_012548/12_Ningguo/result.json count_based_adaptation          None            None   63.74%
   ablation_memory_count_based_v2_20260723_012548/01_Dangtu/result.json count_based_adaptation          None            None   60.03%
   ablation_memory_count_based_v2_20260723_012548/14_Lishui/result.json count_based_adaptation          None            None   59.89%
  ablation_memory_count_based_v2_20260723_012548/13_Gaochun/result.json count_based_adaptation          None            None   51.84%
   ablation_memory_count_based_v2_20260723_012548/05_Suncun/result.json count_based_adaptation          None            None   72.49%
     ablation_memory_count_based_v2_20260723_012548/02_Wuhu/result.json count_based_adaptation          None            None   69.33%
 ablation_memory_count_based_v2_20260723_012548/08_Jingxian/result.json count_based_adaptation          None            None   54.66%
 ablation_memory_count_based_v2_20260723_012548/09_Fanchang/result.json count_based_adaptation          None            None   70.85%
```

**Finding**: Checked all trained checkpoints in `output/0722`. All main models use default fixed parameters `num_slots=4` ($M=4$) and `write_top_k=1` ($K=1$).

---

## 5. Retrained Missing Experiments Audit (100% Completed on 2026-07-24)

All 30 missing experiment runs across Prompt Design variants and Hyperparameter Sweeps ($M$ and $K$) have been trained from scratch on GPU (CUDA) and verified against raw `result.json` outputs.

**Source Directory**: `output/0722/missing_ablations_20260724/missing_ablations_summary.json`

### Verified Prompt Design & Composition Ablation (Table 4)
| Setting | Qingyang | Jingxian | Chizhou | 3-Region Avg |
| :--- | :---: | :---: | :---: | :---: |
| **Query Prompt Construction** | | | | |
| Acoustic Query (w/o RPL) | 59.30% | 46.87% | 61.95% | 56.04% |
| Single Prompt ($L=1$) | 64.37% | 47.42% | 67.54% | 59.78% |
| Uniform Routing | 64.72% | 48.70% | 67.18% | 60.20% |
| Dynamic Routing (Full $L=8$) | 61.44% | 49.60% | 63.73% | 58.26% |
| **Unseen-Region Source-Prompt Composition** | | | | |
| Uniform Composition | 64.72% | 48.70% | 67.18% | 60.20% |
| Single Source Prompt | 64.37% | 49.95% | 65.93% | 60.08% |
| Unconstrained Composition | 64.42% | 47.94% | 67.70% | 60.02% |
| Convex Composition (Full) | 61.44% | 49.60% | 63.73% | 58.26% |

### Verified Persistent Memory Hyperparameter Sweeps (Table 6)
| Parameter / Setting | Qingyang | Jingxian | Chizhou | 3-Region Avg |
| :--- | :---: | :---: | :---: | :---: |
| **Memory Slots per Class ($M$)** | | | | |
| $M=1$ | 64.53% | 51.30% | 65.78% | 60.54% |
| $M=2$ | 65.50% | 49.89% | 68.08% | 61.16% |
| $M=4$ (Default) | 61.44% | 49.60% | 63.73% | 58.26% |
| $M=8$ | 63.97% | 47.27% | 66.41% | 59.22% |
| **Top-$K$ Write Mechanism ($K$)** | | | | |
| $K=1$ (Default) | 61.44% | 49.60% | 63.73% | 58.26% |
| $K=2$ | 64.42% | 47.96% | 67.72% | 60.03% |
| $K=4$ | 64.42% | 48.00% | 67.75% | 60.06% |

