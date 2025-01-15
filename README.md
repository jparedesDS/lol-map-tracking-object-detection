# League of Legends Map Tracking Object Detection
Detection of League of legends player in map with AI (Yolo11)

#### Supported Labels
['Aatrox', 'Ahri', 'Akali', 'Akshan', 'Alistar', 'Amumu', 'Anivia', 'Annie', 'Aphelios', 'Ashe', 'AurelionSol', 'Azir', 'Bard', 'Blitzcrank', 'Brand', 'Braum', 'Caitlyn', 
'Camille', 'Cassiopeia', 'ChoGath', 'Chogath', 'Corki', 'Darius', 'Diana', 'DrMundo', 'Draven', 'Ekko', 'Elise', 'Evelynn', 'Ezreal', 'Fiddlesticks', 'Fiora', 'Fizz', 'Galio', 
'Gangplank', 'Garen', 'Gnar', 'Gragas', 'Graves', 'Gwen', 'Hecarim', 'Heimerdinger', 'Illaoi', 'Irelia', 'Ivern', 'Janna', 'Jarvan', 'Jax', 'Jayce', 'Jaycew', 'Jhin', 'Jinx', 
'Kaisa', 'Kalista', 'Kane', 'Karma', 'Karthus', 'Kassadin', 'Katarina', 'Kayle', 'Kayn', 'Kennen', 'KhaZix', 'Kindred', 'Kled', 'KogMaw', 'LeBlanc', 'LeeSin', 'Leona', 'Lilia', 
'Lillia', 'Lissandra', 'Lucian', 'Lulu', 'Lux', 'Malphite', 'Malzahar', 'Maokai', 'MasterYi', 'Mindred', 'MissFortune', 'Mordekaiser', 'Morgana', 'Nami', 'Nasus', 'Nautilus', 
'Neeko', 'Nidalee', 'Nilah', 'Nocturne', 'Nunu', 'Olaf', 'Orianna', 'Ornn', 'Pantheon', 'Poppy', 'Pyke', 'Qiyana', 'Quinn', 'Rakan', 'Rammus', 'RekSai', 'Reksai', 'Rell', 'Renata', 
'Renekton', 'Rengar', 'Riven', 'Rumble', 'Ryze', 'Samira', 'Sejuani', 'Senna', 'Seraphine', 'Sett', 'Shaco', 'Shen', 'Shyvana', 'Singed', 'Sion', 'Sivir', 'Skarner', 'Sona', 
'Soraka', 'Swain', 'Sylas', 'Syndra', 'TahmKench', 'Taliyah', 'Talon', 'Taric', 'Teemo', 'Thresh', 'Tristana', 'Trundle', 'Tryndamere', 'TwistedFate', 'Twitch', 'Udyr', 'Urgot', 
'Varus', 'Vayne', 'Veigar', 'VelKoz', 'Velkoz', 'Vex', 'Vi', 'Viego', 'Viktor', 'Vladimir', 'Volibear', 'Warwick', 'Wukong', 'Xayah', 'Xerath', 'XinZhao', 'Yasuo', 'Yone', 
'Yorick', 'Yuumi', 'Zac', 'Zed', 'Zeri', 'Ziggs', 'Zilean', 'Zoe', 'Zyra']

#### ALL my models YOLO11, YOLOv10 & YOLOv9
- Yolov9c: https://huggingface.co/jparedesDS/cs2-yolov9c
- Yolov10s: https://huggingface.co/jparedesDS/cs2-yolov10s
- Yolov10m: https://huggingface.co/jparedesDS/cs2-yolov10m
- Yolov10b: https://huggingface.co/jparedesDS/cs2-yolov10b
- Yolov10b: https://huggingface.co/jparedesDS/valorant-yolov10b
- Yolo11x: https://huggingface.co/jparedesDS/welding-defects-detection

#### How to use
```
from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO(r'weights\yolo11m_lol-tracking.pt')

# Run inference on 'image.png' with arguments
model.predict(
    'image.png',
    save=True,
    device=0
    )
```
#### Confusion matrix normalized
![confusion_matrix_normalized.png](https://cdn-uploads.huggingface.co/production/uploads/62e1c9b42e4cab6e39dafc97/Bu6XXKVGDh-PfEq1HYVSC.png)
#### Labels
![labels.jpg](https://cdn-uploads.huggingface.co/production/uploads/62e1c9b42e4cab6e39dafc97/fuHk2Tfi_kV6wBGKFuECC.jpeg)
#### Results
![results.png](https://cdn-uploads.huggingface.co/production/uploads/62e1c9b42e4cab6e39dafc97/U4993AuuhClKD-eETn6kN.png)
#### Predict
![val_batch0_labels.jpg](https://cdn-uploads.huggingface.co/production/uploads/62e1c9b42e4cab6e39dafc97/DxSk38N_EhXjHjgdy1g_8.jpeg)
![val_batch1_labels.jpg](https://cdn-uploads.huggingface.co/production/uploads/62e1c9b42e4cab6e39dafc97/CWsVKQ2RUx2cBhDCNMExI.jpeg)
![val_batch1_pred.jpg](https://cdn-uploads.huggingface.co/production/uploads/62e1c9b42e4cab6e39dafc97/Jh0WFSbpXUEGXCXozQV78.jpeg)
```
YOLO11m summary (fused): 303 layers, 20,158,789 parameters, 0 gradients, 68.4 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 9/9 [00:05<00:00,  1.67it/s]
                   all        822       1058      0.978      0.991      0.993       0.97
                Aatrox          5          5      0.977          1      0.995      0.995
                  Ahri          6          6      0.981          1      0.995      0.995
                 Akali          4          4      0.979          1      0.995      0.995
               Alistar          7          7      0.983          1      0.995      0.995
                 Amumu          3          3          1          1      0.995      0.995
                Anivia          6          6      0.983          1      0.995      0.995
                 Annie         21         21      0.998          1      0.995      0.875
              Aphelios          4          4          1          1      0.995      0.995
                  Ashe          1          1          1          1      0.995      0.995
           AurelionSol          4          4      0.974          1      0.995      0.995
                  Azir          8          8      0.997          1      0.995      0.856
                  Bard         22         22          1      0.975      0.995      0.874
            Blitzcrank          8          8      0.985          1      0.995      0.995
                 Brand          3          3      0.962          1      0.995      0.995
                 Braum          5          5      0.984          1      0.995      0.995
               Caitlyn          5          5      0.985          1      0.995       0.93
               Camille          7          7      0.989          1      0.995      0.875
            Cassiopeia          3          3      0.988          1      0.995       0.74
               Chogath          3          3      0.963          1      0.995      0.995
                 Corki          7          7      0.986          1      0.995      0.995
                Darius          3          3      0.966          1      0.995      0.995
                 Diana          7          7      0.984          1      0.995      0.995
               DrMundo          4          4      0.973          1      0.995      0.995
                Draven          3          3      0.963          1      0.995      0.995
                  Ekko          8          8       0.99          1      0.995      0.953
                 Elise          4          4      0.975          1      0.995      0.995
               Evelynn          5          5      0.978          1      0.995      0.995
                Ezreal          6          6      0.981          1      0.995      0.995
          Fiddlesticks          9          9      0.988          1      0.995      0.995
                 Fiora          3          3      0.967          1      0.995      0.995
                  Fizz          6          6      0.981          1      0.995      0.995
                 Galio          5          5      0.978          1      0.995      0.995
             Gangplank          5          5       0.98          1      0.995      0.995
                 Garen          4          4      0.997          1      0.995      0.995
                  Gnar          3          3      0.966          1      0.995      0.995
                Gragas          5          5      0.989          1      0.995       0.89
                Graves         14         14          1      0.837      0.937      0.852
                  Gwen         11         11      0.995          1      0.995      0.947
               Hecarim         13         13          1      0.932      0.995      0.954
          Heimerdinger          8          8        0.8          1      0.995      0.893
                Illaoi          5          5      0.978          1      0.995      0.995
                Irelia         16         17      0.998      0.941      0.979      0.807
                 Ivern          4          4      0.973          1      0.995      0.995
                 Janna          7          7      0.994          1      0.995      0.995
                Jarvan          3          3      0.974          1      0.995      0.995
                   Jax          7          7      0.984          1      0.995      0.995
                 Jayce          2          2      0.956          1      0.995      0.995
                  Jhin          4          4      0.981          1      0.995      0.949
                  Jinx          4          4      0.973          1      0.995      0.995
                 Kaisa         23         23      0.955      0.927       0.99      0.879
               Kalista          7          7      0.997          1      0.995      0.995
                  Kane          6          6      0.983          1      0.995      0.995
                 Karma         12         12      0.984          1      0.995      0.886
               Karthus          5          5       0.98          1      0.995      0.995
              Kassadin          1          1      0.933          1      0.995      0.995
              Katarina          5          5      0.987          1      0.995      0.911
                 Kayle          9          9      0.987          1      0.995      0.995
                  Kayn          9          9       0.99          1      0.995      0.986
                Kennen          5          5          1          1      0.995      0.995
                KhaZix          7          7      0.985          1      0.995      0.995
                  Kled          5          5      0.975          1      0.995      0.995
                KogMaw          7          7      0.983          1      0.995      0.995
               LeBlanc          8          8      0.986          1      0.995      0.995
                LeeSin          3          3      0.967          1      0.995      0.995
                 Leona          9          9      0.987          1      0.995      0.995
                Lillia          4          4      0.976          1      0.995      0.995
             Lissandra          6          6      0.987          1      0.995      0.982
                Lucian         17         17          1      0.885      0.987      0.902
                  Lulu          3          3      0.966          1      0.995      0.995
                   Lux          3          3      0.969          1      0.995      0.995
              Malphite         26         26          1      0.995      0.995      0.902
              Malzahar          3          3      0.968          1      0.995      0.995
                Maokai          4          4       0.97          1      0.995      0.995
              MasterYi          2          2      0.957          1      0.995      0.995
               Mindred          7          7      0.985          1      0.995      0.995
           MissFortune          4          4      0.976          1      0.995      0.995
           Mordekaiser          7          7      0.988          1      0.995      0.995
               Morgana          6          6          1          1      0.995      0.995
                  Nami          4          4      0.975          1      0.995      0.995
                 Nasus          6          6       0.98          1      0.995      0.995
              Nautilus          9          9          1      0.938      0.995       0.94
                 Neeko          5          5      0.979          1      0.995      0.995
               Nidalee          5          5      0.982          1      0.995      0.995
                 Nilah          3          3      0.662      0.667       0.83      0.558
              Nocturne          6          6       0.98          1      0.995      0.995
                  Nunu          3          3      0.972          1      0.995      0.995
                  Olaf          3          3      0.968          1      0.995      0.995
               Orianna          8          8      0.986          1      0.995      0.995
                  Ornn          8          8      0.992          1      0.995       0.92
              Pantheon          1          1      0.928          1      0.995      0.995
                 Poppy          3          3      0.966          1      0.995      0.995
                Qiyana          5          5      0.976          1      0.995      0.995
                 Quinn          4          4      0.972          1      0.995      0.995
                 Rakan          3          3      0.964          1      0.995      0.995
                Rammus          5          5      0.977          1      0.995      0.995
                Reksai          1          1      0.931          1      0.995      0.995
                  Rell          4          4      0.972          1      0.995      0.995
                Renata          6          6      0.988          1      0.995      0.936
              Renekton          8          8      0.991          1      0.995      0.955
                Rengar          3          3      0.966          1      0.995      0.995
                 Riven          6          6      0.984          1      0.995      0.995
                Rumble          1          1      0.917          1      0.995      0.995
                  Ryze          6          6      0.982          1      0.995      0.995
                Samira          8          8      0.987          1      0.995      0.962
               Sejuani          8          8      0.988          1      0.995      0.952
                 Senna          6          6      0.999          1      0.995      0.995
             Seraphine         24         24      0.957      0.928      0.987      0.889
                  Sett          6          6      0.981          1      0.995      0.995
                 Shaco          6          6      0.981          1      0.995      0.995
                  Shen          3          3      0.969          1      0.995      0.995
               Shyvana          3          3      0.967          1      0.995      0.995
                Singed          7          7      0.991          1      0.995      0.912
                  Sion         23         23      0.998          1      0.995      0.881
                 Sivir          7          7          1          1      0.995      0.995
               Skarner          5          5       0.98          1      0.995      0.995
                  Sona          6          6      0.981          1      0.995      0.995
                Soraka          7          7      0.992          1      0.995      0.995
                 Swain          4          4      0.977          1      0.995      0.995
                 Sylas         20         20          1      0.877      0.985      0.823
                Syndra         21         21          1      0.957      0.995      0.866
             TahmKench          7          7      0.988          1      0.995      0.995
               Taliyah          7          7      0.984          1      0.995      0.995
                 Talon          7          7      0.992          1      0.995      0.934
                 Taric          5          5      0.977          1      0.995      0.995
                 Teemo          4          4      0.974          1      0.995      0.995
                Thresh          7          7          1      0.876      0.995      0.995
              Tristana          7          7      0.995          1      0.995      0.957
               Trundle          5          5      0.978          1      0.995      0.995
            Tryndamere          7          7      0.983          1      0.995      0.995
           TwistedFate          4          4      0.976          1      0.995      0.995
                Twitch         10         10      0.988          1      0.995      0.995
                  Udyr          7          7      0.981          1      0.995       0.96
                 Urgot          7          7      0.985          1      0.995      0.995
                 Varus          5          5          1          1      0.995      0.995
                 Vayne          3          3      0.968          1      0.995      0.995
                Veigar          3          3      0.963          1      0.995      0.995
                Velkoz          5          5          1          1      0.995      0.995
                   Vex          5          5      0.977          1      0.995      0.995
                    Vi          5          5      0.973          1      0.995      0.995
                 Viego          6          6          1      0.997      0.995      0.924
                Viktor         13         13      0.996          1      0.995      0.978
              Vladimir         15         15      0.997          1      0.995      0.957
              Volibear          6          6      0.983          1      0.995      0.995
               Warwick          3          3      0.972          1      0.995      0.995
                Wukong         10         10      0.993          1      0.995       0.94
                 Xayah          6          6       0.98          1      0.995      0.995
                Xerath          2          2      0.953          1      0.995      0.995
               XinZhao          6          6      0.982          1      0.995      0.995
                 Yasuo          6          6      0.989          1      0.995       0.92
                  Yone         12         12          1      0.972      0.995       0.95
                Yorick          9          9      0.987          1      0.995      0.995
                 Yuumi          9          9          1      0.983      0.995      0.891
                   Zac          6          6      0.981          1      0.995      0.995
                   Zed          3          3       0.97          1      0.995      0.995
                  Zeri          4          4      0.973          1      0.995      0.995
                 Ziggs          3          3      0.965          1      0.995      0.995
                Zilean         10         10          1      0.899      0.995      0.917
                   Zoe         13         13      0.996          1      0.995       0.96
                  Zyra          2          2      0.958          1      0.995      0.995
```

#### Others models...
https://huggingface.co/jparedesDS/
