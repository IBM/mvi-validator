# Maximo Visual Inspection (MVI) Validator



[MVI Validator](https://github.com/IBM/mvi-validator) is an accuracy validator for Maximo Visual Inspection.



## Setup （セットアップ）

1. Pythonをインストールする。
    コマンドラインから`python3`コマンドと`pip3`コマンドを実行できるか確認する
    
    ```sh
    $ python3 --version
    Python 3.11.0
    
    $ pip3 --version 
    pip 22.3 from pip (python 3.11)
    ```

3. インストール
    ```bash
    $ git clone git@github.com:IBM/mvi-validator.git
    $ cd mvi-validator
    $ pip3 install -e .
    
    ```
    
    コマンドラインから`mvi-validator`コマンドを実行できるか確認する。
    ```sh
    $ mvi-validator --version
    0.0.3
    
    # Command not found エラーが出る場合は、`mvi-validator` を `python -m mvi-validator` にすると動くかもしれません
    $ python -m mvi-validator --version
    0.0.3
    ```



## Usage（使い方）

1. MVI > 左メニュー > データセット > テスト用に作成したデータセットを選択 > 右上の エクスポートボタン![image-20230118112342839](README.assets/image-20230118112342839.png)をクリック > zipファイルをローカルPCに保存
    ![image-20230118112524488](README.assets/image-20230118112524488.png)

    

2. ダウンロードしたzipファイルを解凍する。
    - 例) カレントディレクトリの下の`test_ball_bearing`ディレクトリに解凍したとすると、以下のようになる
        ```
        $ tree . | head
        .
        └── test_ball_bearing
            ├── 00196b51-d6e7-4372-81a6-f99f15541520.jpg
            ├── 00196b51-d6e7-4372-81a6-f99f15541520.xml
            ├── 002dc8fb-1806-4f73-9df8-4e93210e08f7.jpg
            ├── 002dc8fb-1806-4f73-9df8-4e93210e08f7.xml
            ├── 004b9976-3b36-4002-861b-d692c7db43dd.jpg
            ├── 004b9976-3b36-4002-861b-d692c7db43dd.xml
            ├── 005d5394-64ea-4d1f-a353-fac0fb5ebcb4.jpg
            ├── 005d5394-64ea-4d1f-a353-fac0fb5ebcb4.xml
        
        ```
    
    
    
3. MVI  > 左メニュー > モデル >  検証したいモデル > デプロイ
   
4. MVI > 左メニュー > デプロイ済みモデル > デプロイ済みモデルのAPIエンドポイントの`コピー`をクリック
    - 例) 以下のページでコピーしてきたURLは `https://mvi.com/api/dlapis/bb44e214-e208-4e6a-a88b-d9ab173023da` になる
        ![image-20230118112109947](README.assets/image-20230118112109947.png)

    

5. ターミナルを起動し、`mvi-validator deployed-model detection --api [APIエンドポイントのURL]  [テストデータのディレクトリ]`を実行する。
   
   
    1. 例) 例えば、APIURLが `https://mvi.com/api/dlapis/bb44e214-e208-4e6a-a88b-d9ab173023da` 、ディレクトリが `test_ball_bearing`の場合
        ```sh
        $ mvi-validator deployed-model detection --api https://mvi.com/api/dlapis/bb44e214-e208-4e6a-a88b-d9ab173023da  test_ball_bearing
        ```
        <img src="README.assets/image-20230118114944241.png" alt="image-20230118114944241" style="zoom:50%;" />
    
    2. 結果はデフォルトでマークダウンで表示される
        ```markdown
        # Summary
        |   num_images |   num_gt_bbox |   num_pd_bbox |   total_tp |   total_fp |   total_fn |   precision |   recall |   f-measure |      mAP | model_id                             |
        |-------------:|--------------:|--------------:|-----------:|-----------:|-----------:|------------:|---------:|------------:|---------:|:-------------------------------------|
        |           28 |            27 |             9 |          8 |          1 |         19 |           1 | 0.185185 |    0.444444 | 0.888889 | bb44e214-e208-4e6a-a88b-d9ab173023da |
        ```
    
        
        
    3. エクセルで表示するとこんな感じ
       ![image-20230118113736188](README.assets/image-20230118113736188.png)
    





## Usage (Jupyter Notebook)



1. Install with notebook option
    ```sh
    $ pip3 install -e '.[notebook]'
    ```
2. Start jupyter notebook
    ```sh
    $ jupyter notebook
    ```







## Contributing

Open Issue [here](https://github.com/IBM/mvi-validator/issues) .



## Authors



Takahide Nogayama

<a href="https://github.com/nogayama"><img src="https://avatars.githubusercontent.com/u/11750755?s=460" width="100"/></a>



## License（ライセンス）

MIT

