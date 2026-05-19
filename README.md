# Tivit_Extension

鏈」鐩槸鍩轰簬 TiViT 鐨勬椂闂村簭鍒楀浘鍍忓寲鍒嗙被瀹為獙鎵╁睍锛屾牳蹇冨叧娉ㄧ偣鏄細**灏嗕竴缁存垨澶氱淮鏃堕棿搴忓垪娓叉煋鎴愭姌绾垮浘鍥惧儚锛屽啀杈撳叆棰勮缁冭瑙?Transformer 鎻愬彇琛ㄧず锛岀敤浜?UCR/UEA 鏃堕棿搴忓垪鍒嗙被瀵规瘮**銆?

閫氳繃缃戠洏鍒嗕韩鐨勬枃浠讹細dataset

閾炬帴: https://pan.baidu.com/s/1ydgwTojNom_kKigBZXd54w?pwd=1234 鎻愬彇鐮? 1234

褰撳墠榛樿杈撳叆褰㈠紡鏄姌绾垮浘锛?

```bash
--image_mode line_plot
```

鍘熷鐨勫垎娈电伆搴﹀浘琛ㄧず浠嶇劧淇濈暀锛屽彲閫氳繃 `--image_mode segment` 鍚敤锛屼富瑕佺敤浜庡拰鍘?TiViT 鏂规鍋氬鐓с€?

## 椤圭洰浜偣

- **鎶樼嚎鍥句紭鍏?*锛氶粯璁ゆ妸姣忔潯鏃堕棿搴忓垪杞崲涓?224 x 224 鐨勭櫧搴曢粦绾垮浘鍍忥紝鏇磋创杩戜汉绫绘煡鐪嬫椂闂村簭鍒楃殑鏂瑰紡銆?
- **鏀寔澶氶€氶亾鏃堕棿搴忓垪**锛氬鍙橀噺鏁版嵁浼氭寜閫氶亾鍒嗗埆缁樺埗鎶樼嚎鍥撅紝鍐嶅垎鍒€佸叆瑙嗚楠ㄥ共缃戠粶鎻愬彇鐗瑰緛銆?
- **澶嶇敤澶ц妯¤瑙夋ā鍨?*锛氭敮鎸?CLIP/OpenCLIP銆丏INOv2銆丼igLIP 2銆丮AE 绛夐璁粌 ViT銆?
- **鍙笌 TSFM 铻嶅悎**锛氭敮鎸佹嫾鎺?Mantis銆丮OMENT 绛夋椂闂村簭鍒楀熀纭€妯″瀷琛ㄧず锛岃瀵熻瑙夎〃绀轰笌鏃堕棿搴忓垪琛ㄧず鐨勪簰琛ユ€с€?
- **淇濈暀鍙鐜板疄楠岃缃?*锛氫娇鐢ㄥ畼鏂?UCR/UEA train/test split锛屽苟灏嗛獙璇侀泦鍒掑垎绱㈠紩淇濆瓨鍒扮粨鏋滅洰褰曘€?

## 鏂规硶姒傝

![Plot](img/Plot.png)

鎶樼嚎鍥炬ā寮忕殑娴佺▼濡備笅锛?

1. 璇诲彇 UCR 鎴?UEA 鏃堕棿搴忓垪鏁版嵁銆?
2. 瀵规瘡鏉″簭鍒楀仛 robust scaling锛屽噺寮卞紓甯稿€煎拰灏哄害宸紓鐨勫奖鍝嶃€?
3. 灏嗗簭鍒楀€煎綊涓€鍖栧埌鍥惧儚鍧愭爣锛屽苟鎻掑€煎埌鍥哄畾瀹藉害銆?
4. 娓叉煋涓虹櫧搴曢粦绾跨殑 RGB 鍥惧儚銆?
5. 杈撳叆鍐荤粨鐨?ViT/CLIP-ViT锛屽湪鎸囧畾灞傛彁鍙?hidden representation銆?
6. 瀵?token 琛ㄧず鍋?`mean` 鎴?`cls_token` 鑱氬悎銆?
7. 浣跨敤 logistic regression銆乶earest centroid 鎴?random forest 瀹屾垚鍒嗙被銆?

鎶樼嚎鍥捐浆鎹㈠疄鐜颁綅浜?[src/tivit.py](src/tivit.py)锛屾牳蹇冨弬鏁版槸 `--image_mode line_plot`銆?

## 鐜瀹夎

寤鸿浣跨敤 Python 3.11銆?

```bash
conda create -n tivit_env python=3.11
conda activate tivit_env
python -m pip install -r requirements.txt
```

## 鏁版嵁闆?

椤圭洰闈㈠悜 [UCR/UEA time series classification benchmark](https://www.timeseriesclassification.com/)銆?

鏁版嵁鍙€氳繃 [aeon](https://www.aeon-toolkit.org) 鍔犺浇銆傞粯璁ゆ祦绋嬩細锛?

- 瀵?UCR 鏁版嵁涓殑缂哄け鍊煎仛绾挎€ф彃鍊硷紱
- 瀵逛笉绛夐暱搴忓垪鍋?padding锛?
- 浠庡畼鏂硅缁冮泦閲屾寜 `--val_ratio` 鍒掑垎楠岃瘉闆嗭紱
- 灏嗗疄闄?train/validation 绱㈠紩鍐欏叆 `result_dir/splits/`锛屾柟渚垮鐜板拰璺ㄦā鍨嬫瘮杈冦€?

濡傞渶浣跨敤 aeon 鑷甫棰勫鐞嗭紝鍙坊鍔狅細

```bash
--aeon
```

## 鏀寔鐨勮瑙夐骞?

| 绫诲瀷 | 绀轰緥 checkpoint |
| --- | --- |
| CLIP/OpenCLIP | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` |
| DINOv2 | `facebook/dinov2-base` |
| SigLIP 2 | `google/siglip2-so400m-patch14-224` |
| MAE | `facebook/vit-mae-base` |

鍏朵腑 CLIP-ViT-H 鏄綋鍓嶆姌绾垮浘瀹為獙涓帹鑽愪紭鍏堜娇鐢ㄧ殑楠ㄥ共銆?

## 鎶樼嚎鍥惧垎绫诲疄楠?

涓嬮潰鏄湪 UCR 鏁版嵁闆嗕笂浣跨敤 CLIP-ViT-H 鎶樼嚎鍥捐緭鍏ヨ繘琛屽垎绫荤殑绀轰緥锛?

```bash
python main.py \
  --vit_1_name laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode line_plot \
  --classifier_type logistic_regression \
  --datasets ucr \
  --data_dir /path/to/your/data \
  --result_dir /path/to/save/results \
  --random_seed 2021 \
  --val_ratio 0.2
```

璇存槑锛?

- `--image_mode line_plot` 鏄粯璁ゅ€硷紝鏄惧紡鍐欏嚭渚夸簬璁板綍瀹為獙閰嶇疆銆?
- 鎶樼嚎鍥炬ā寮忎笉闇€瑕?`--patch_size`锛屼唬鐮佷細鑷姩璺宠繃 2D 鍒嗘 patch size 鎼滅储銆?
- `--vit_1_layer 14` 琛ㄧず鎴彇 ViT 鐨勭 14 灞傝〃绀恒€?
- `--aggregation mean` 琛ㄧず瀵?token hidden states 鍋氬钩鍧囨睜鍖栥€?

## 鎸囧畾鏁版嵁闆嗚繍琛?

鍙互鐢?`--dataset_names` 鍙窇閮ㄥ垎鏁版嵁闆嗐€備笅闈㈡槸鍜?DMMV 瀵规瘮鏃跺父鐢ㄧ殑鍙楁帶璁剧疆锛?

- random seed: `2021`
- validation ratio: `0.2`
- UCR: `ECG200`, `FordA`
- UEA: `BasicMotions`, `SelfRegulationSCP1`

杩愯 UCR 鎶樼嚎鍥惧疄楠岋細

```bash
python main.py \
  --vit_1_name laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode line_plot \
  --classifier_type logistic_regression \
  --datasets ucr \
  --dataset_names ECG200 FordA \
  --data_dir /path/to/your/data \
  --result_dir /path/to/save/results \
  --random_seed 2021 \
  --val_ratio 0.2
```

杩愯 UEA 鎶樼嚎鍥惧疄楠岋細

```bash
python main.py \
  --vit_1_name laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode line_plot \
  --classifier_type logistic_regression \
  --datasets uea \
  --dataset_names BasicMotions SelfRegulationSCP1 \
  --data_dir /path/to/your/data \
  --result_dir /path/to/save/results \
  --random_seed 2021 \
  --val_ratio 0.2
```

椤圭洰涔熸彁渚涗簡鑴氭湰鍏ュ彛锛?

```bash
bash scripts/run_lineplot_ucr.sh
bash scripts/run_lineplot_uea.sh
```

## 涓庢椂闂村簭鍒楀熀纭€妯″瀷铻嶅悎

濡傛灉甯屾湜姣旇緝鎴栬瀺鍚堜紶缁熸椂闂村簭鍒楀熀纭€妯″瀷锛屽彲浠ユ坊鍔狅細

```bash
--mantis
```

鎴栵細

```bash
--moment base
```

渚嬪锛屽皢鎶樼嚎鍥?ViT 琛ㄧず鍜?Mantis 琛ㄧず鎷兼帴鍚庡垎绫伙細

```bash
python main.py \
  --vit_1_name laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --vit_1_layer 14 \
  --aggregation mean \
  --image_mode line_plot \
  --mantis \
  --classifier_type logistic_regression \
  --datasets ucr \
  --data_dir /path/to/your/data \
  --result_dir /path/to/save/results \
  --random_seed 2021 \
  --val_ratio 0.2
```

## 缁撴灉鏂囦欢

姣忔杩愯浼氬湪 `result_dir` 涓嬫柊寤哄甫鏃堕棿鎴崇殑瀹為獙鐩綍锛岄€氬父鍖呭惈锛?

- `args.json`锛氭湰娆¤繍琛岀殑瀹屾暣鍙傛暟锛?
- `train_val.csv`锛氬悇鏁版嵁闆嗛獙璇侀泦鍜屾祴璇曢泦缁撴灉锛?
- `splits/*.npz`锛氫粠瀹樻柟璁粌闆嗗垝鍒嗗嚭鐨?train/validation 绱㈠紩銆?

缁撴灉琛ㄤ腑浼氳褰?`image_mode`锛屽洜姝ゅ彲浠ョ洿鎺ュ尯鍒嗘姌绾垮浘杈撳叆鍜屽垎娈电伆搴﹀浘杈撳叆銆?

## 鍏朵粬鍒嗘瀽鍔熻兘

闄や簡鍒嗙被锛屾湰椤圭洰杩樹繚鐣?TiViT 鐨勮〃绀哄垎鏋愬姛鑳斤細

```bash
--get_intrinsic_dimension
--get_principal_components
--measure_alignment
```

杩欎簺鍔熻兘鍙敤浜庡垎鏋愪笉鍚?ViT 灞傜殑琛ㄧず缁撴瀯锛屼互鍙婃姌绾垮浘瑙嗚琛ㄧず鍜?Mantis/MOMENT 琛ㄧず涔嬮棿鐨?alignment銆?

## 涓昏鍙傛暟

| 鍙傛暟 | 浣滅敤 |
| --- | --- |
| `--image_mode line_plot` | 浣跨敤鎶樼嚎鍥句綔涓鸿瑙夎緭鍏ワ紝褰撳墠榛樿妯″紡 |
| `--image_mode segment` | 浣跨敤鍘?TiViT 鍒嗘鐏板害鍥捐緭鍏?|
| `--vit_1_name` | 绗竴涓瑙夐骞?checkpoint |
| `--vit_1_layer` | 鎻愬彇琛ㄧず鐨?ViT 灞?|
| `--aggregation` | hidden states 鑱氬悎鏂瑰紡锛屾敮鎸?`mean` 鍜?`cls_token` |
| `--classifier_type` | 鍒嗙被鍣紝鏀寔 `logistic_regression`銆乣nearest_centroid`銆乣random_forest` |
| `--datasets` | 閫夋嫨 `ucr` 鎴?`uea` |
| `--dataset_names` | 闄愬畾瑕佽繍琛岀殑鏁版嵁闆嗗悕绉?|
| `--mantis` | 鎷兼帴 Mantis 鏃堕棿搴忓垪琛ㄧず |
| `--moment` | 鎷兼帴 MOMENT 琛ㄧず锛屾敮鎸?`small`銆乣base`銆乣large` |

瀹屾暣鍙傛暟瀹氫箟瑙?[src/arguments.py](src/arguments.py)銆?

## 鑷磋阿

鏈」鐩熀浜?TiViT 鎬濊矾鎵╁睍鎶樼嚎鍥捐緭鍏ュ疄楠岋紝骞跺弬鑰冧簡浠ヤ笅宸ヤ綔锛?

- TiViT: Time Series Representations for Classification Lie Hidden in Pretrained Vision Transformers
- OpenCLIP / CLIP 瑙嗚楠ㄥ共
- Mantis 鍜?MOMENT 鏃堕棿搴忓垪鍩虹妯″瀷
- aeon 鏃堕棿搴忓垪鏁版嵁闆嗗伐鍏?
