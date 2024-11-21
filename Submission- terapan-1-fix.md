

# Laporan Proyek Machine Learning

## Domain Proyek
Domain yang dipilih untuk prediksi adalah kesehatan, dengan fokus pada penyakit diabetes. 

Diabetes melitus merupakan salah satu penyakit autoimun yang disebabkan oleh berbagai faktor, termasuk genetika, lingkungan, pola makan,serta gaya hidup. Hong Sun, peneliti dari International Diabetes Federation, Brussels, Belgium dalam tulisannya yang berjudul [IDF Diabetes Atlas: Global, regional and country-level diabetes prevalence estimates for 2021 and projections for 2045](https://www.sciencedirect.com/science/article/abs/pii/S0168822721004782?casa_token=PUqaKleRbzQAAAAA:96sPp6aNS4T_akTWh7_WPwxDBadO-wGLcqURVJdkTcyb5ggj6y1VDsidDtddFuNJwh6tnU4ZsQ), menyebutkan bahwa nilai prevalensi estimasi pada rentang usia 20–79 tahun pada 2021 adalah 10.5% (536.6 juta orang). Diprediksikan pula akan terjadi peningkatan sebesar 12.2% (783.2 juta) pada tahun 2045.

Kondisi ini patut menjadi perhatian mengingat jika tidak ditangani, diabetes dapat merusak sistem saraf, pembuluh darah, mata, jantung, ginjal, dan berpotensi menyebabkan stroke, amputasi anggota tubuh bagian bawah, serta akhirnyadapat menyebabkan kematian. Kondisi ini diperparah dengan belum ditemukannya obat untuk menyembuhkan diabetes selain mengendalikan kadar glukosa dalam darah.  

Oleh karena itu, tindakan pencegahan penyakit diabetes sejak dini menjadi langkah penting untuk menurunkan tingkat mortalitas. Dengan data pasien yang menganduk faktor-faktor pendorong munculnya penyakit Diabetes seperti kehamilan,tingkat glukosa dalam darah, tekanan darah tinggi,serta diagnosis yang diberikan, akan dilakukan analisa prediktif menggunakan machine learning yang dapat digunakan sebagai dasar prefentif penyakit diabetes.  

Referensi:
[1] [A risk assessment and prediction framework for diabetes mellitus using
machine learning algorithms](https://www.sciencedirect.com/science/article/pii/S2772442523001405)

## Business Understanding

### Problem Statements
Bagaimana mengetahui variabel-variabel kesehatan yang berpotensi mempengaruhi apakah pasien memiliki potensi penyakit diabetes (_diabetic disease_)?

### Goals
Untuk mengetahui kecenderungan (_prediksi_) suatu pasien menderita penyakit diabetes berdasarkan rvariabel kesehatannya.

### Solution statements
Solusi pembuatan model yang dilakukan adalah dengan menerapkan 4 algoritma machine learning, terbatas pada **_K-NN_**, **_Random Forest_**, **_Logistic Regression_**, dan **_SVM_**. Diterapkannya 4 algoritma tersebut bertujuan untuk mengkomparasi dan mendapatkan model atau algoritma yang memiliki tingkat _error_ yang paling kecil, sehingga prediksi penyakit jantung memiliki akurasi yang tinggi.

## Data Understanding
Dataset yang digunakan pada proyek _machine learning_ merupakan data yang didapatkan dari situs yang didapat dari situs [kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset). Dataset ini merupakan kumpulan data medis dan demografis dari pasien, beserta status diabetes mereka . Data ini mencakup fitur-fitur seperti **usia**, **jenis kelamin**, **indeks massa tubuh** (***BMI***), **hipertensi**, **penyakit jantung**, **riwayat merokok**, **kadar HbA1c**, dan **kadar glukosa darah**.


> **Variabel-variabel pada Diabetic Dataset adalah sebagai berikut:**

1.  Age: usia pasien, dalam tahun (_years_)
2.  gender: jenis kelamin pasien
	- M: Pria (_Male_),
	- F: Wanita (_Female_)
3.  Indeks Massa Tubuh (kg/m²)
    - Indeks Massa Tubuh (BMI) merupaka ukuran lemak tubuh yang dihitung berdasarkan perbandingan antara berat dan tinggi badan seseorang.
    - Tingginya nilai BMI seseorang berhubungan dengan risiko tingginya risiko diabetes tipe 2 serta meningkatkan resistansi terhadap insulin sehingga regulasi kadar gula darah terganggu.
    - Variable :    
      - Nilai Numerik antara 10-95.69

4.  Hypertensi: tekanan darah tinggi (_mm Hg_)
    - Tenanan darah tinggi erat hubungannya dengan diabetes.Peningkatan kadar insulin (*hiperinsulinemia*), yang umum terjadi pada tahap awal resistensi insulin, dapat menyebabkan retensi natrium pada ginjal dan akumulasi cairan, yang mengakibatkan peningkatan volume darah dan dapat mengakibatkan hipertensi.
    - Variable :    
      - 0 : (_Normal_)
      - 1 : (_Hipertensi_)
5.  Penyakit Jantung:
    - Penyakit jantung, termasuk kondisi seperti penyakit arteri koroner dan gagal jantung, terkait dengan peningkatan risiko diabetes. Hubungan antara penyakit jantung dan diabetes bersifat dua arah, yang berarti memiliki salah satu kondisi tersebut meningkatkan risiko berkembangnya kondisi lainnya. Hal ini disebabkan oleh adanya faktor risiko yang sama, seperti obesitas, tekanan darah tinggi, dan kolesterol tinggi.
    - Variable :    
      - 0 : (_Normal_)
      - 1 : (_Hipertensi_)

6.  Riwayat Merokok:
    - Merokok dapat mempengaruhi metabolisme glukosa secara langsung dengan mengganggu fungsi sel beta pankreas, yang memproduksi insulin. Gangguan ini dapat menyebabkan penurunan sekresi insulin sebagai respons terhadap glukosa.

    - Variable :
      - Never
      - No Info
      - Current
      - Former
      - ever
      - not current

7.  kadar HbA1c (%):
    - Kadar HbA1c adalah ukuran rata-rata kadar glukosa dalam darah selama periode tiga bulan terakhir. HbA1c (hemoglobin A1c) terbentuk ketika glukosa dalam darah mengikat hemoglobin, protein dalam sel darah merah yang mengangkut oksigen.
    - Nilai HbA1c biasanya dinyatakan dalam persen (%), dengan kadar di bawah 5,7% dianggap normal, antara 5,7% hingga 6,4% menunjukkan pradiabetes, dan 6,5% atau lebih tinggi menunjukkan diabetes.
    - Variable :    
      - Nilai Numerik antara 3.5 - 9
      
8.   Kadar glukosa darah (mg/dL)
    - Kadar glukosa darah mengacu pada jumlah glukosa (gula) yang terdapat dalam darah pada waktu tertentu. Kadar glukosa darah yang tinggi, terutama dalam keadaan puasa atau setelah mengonsumsi karbohidrat, dapat mengindikasikan gangguan regulasi glukosa dan meningkatkan risiko perkembangan diabetes. Pemantauan kadar glukosa darah secara rutin sangat penting dalam diagnosis dan pengelolaan diabetes.
     - Variable :
     	- Nilai Numerik antara 80 dan 300
        
10.   Diabetes
    - Fitur diabetes menggambarkan apakah pasien diketahui memiliki diabetes atau tidak. Fitur ini menjadi target dari perhitungan.
      - Variable :
     	- 0 : (_Normal_)
      	- 1 : (_Diabetes_)

Secara keseluruhan, data memiliki 100000 entries, 0 to 99999. Data juga memiliki 9 kolom yang terdiri dari:
| #  | Column              | Non-Null Count   | Dtype    |
|----|---------------------|------------------|----------|
| 0  | gender              | 100000 non-null | object   |
| 1  | age                 | 100000 non-null | float64  |
| 2  | hypertension        | 100000 non-null | int64    |
| 3  | heart_disease       | 100000 non-null | int64    |
| 4  | smoking_history     | 100000 non-null | object   |
| 5  | bmi                 | 100000 non-null | float64  |
| 6  | HbA1c_level         | 100000 non-null | float64  |
| 7  | blood_glucose_level | 100000 non-null | int64    |
| 8  | diabetes            | 100000 non-null | int64    |


Diketahui data memiliki 3854 data terduplikasi dari 9 fitur yang tersedia.

| Metric                         | Count                              |
|--------------------------------|------------------------------------|
| Number of duplicate rows       | (3854, 9)                         |


Kemudian dilakukan pengecekkan _distinct value_ dari 9 kolom tersebut dan didapatkan : 

| Feature              | Distinct Values |
|-----------------------|-----------------|
| Gender               | 3               |
| Age                  | 102             |
| Hypertension         | 2               |
| Heart Disease        | 2               |
| Smoking History      | 6               |
| BMI                  | 4247            |
| HbA1c Level          | 18              |
| Blood Glucose Level  | 18              |
| Diabetes             | 2               |



[2]: [Diabetic Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

### Explanatory Data Analysis
Untuk memahami data _diabetic desise_ dilakukan visualisasi menggunakan _bar chart_ dan _pie chart_. Dalam memvisualisasikannya, dilakukan dengan _Univariate Analysis_, _Bivariate Analysis_, dan _Multivariate Analysis_. Untuk keseluruhannya, dataset dibagi menjadi dua fitur, yakni fitur _categorical_ dan fitur _numerical_.

#### *Univariate Analysis - Categorical Feature*

1. _Univariate Analysis_ terhadap Diabetic Disease

![1  countplot diabetes](https://github.com/user-attachments/assets/5b64ee66-f6d2-4e5d-9ef9-c47ae4464cf0)

Perlu diketahui bahwa variabel target dari predictive analysis yang dilakukan adalah diabetes.

Dari plot yang dibuat, dapat diketahui bahwa data lebih banyak menunjukkan kondisi diabetes [1] dibanding kondisi normal [0].

2. _Univariate Analysis_ terhadap gender

![2  countplot gender](https://github.com/user-attachments/assets/f5d8e682-ed9c-478c-8117-32dfb836e0f7)


Menampilkan jumlah frekuensi data gender (Male and Female). Datap diketahu bahwa distribusi pasien wanita lebih banyak dari pada pasian pria.

3. _Univariate Analysis_ terhadap _Smoking History_

![3  distribusi rokok](https://github.com/user-attachments/assets/4ca476de-86e1-40eb-bce8-5ce15e87cc67)

Countplot Menampilkan jumlah frekuensi data smoking history. Dapat diketahui dari grafik bahwa data 'No Info' memiliki jumlah paing banyak. Jumlah paling banyak ke dua diduduki olehh pasien yang tdak pernah merokok dengan label never.

4. _Univariate Analysis_ terhadap _Hypertension_

![ua-hypertension](https://github.com/user-attachments/assets/788efa11-0279-475d-adf9-277ab87cceda)


Grafik menampilkan distribusi pasien dengan riwayat tekanan darah tinggi (_hypertension_). Didapatkan 92.2% pasien tidak menderita tekanan darah tinggi.

5. _Univariate Analysis_ terhadap sakit jantung

![ua-heart disease](https://github.com/user-attachments/assets/cadf284a-98e4-4f25-8bf8-d2bd6369e2fd)

Grafik menampilkan distribusi pasien dengan riwayat sakit jantung (heart disease). Didapatkan 95.9% pasien tidak menderita sakit jantung.

#### *Bivariate Analysis - Categorical Feature*
1. Diabetes vs gender

![ba-diabetes vs gender](https://github.com/user-attachments/assets/38355d7a-97a3-4bc8-9e46-a2b1b706aa7f)


Mengecek hubungan antara penderita diabetes dan gender. didapatkan 4035 penderita diabetes laki- laki, yakni 11.2% dari total pasien laki-laki. Didapatkan pula 4447 penderita diabetes wanita, yakni 8.6% dari total pasien wanita.

2. Diabetes vs smoking history

![ba-diabetes vs smoking history](https://github.com/user-attachments/assets/d8fd1ff9-3033-4814-87c4-e4c8829dc471)


Pada fitur Smoking History, penderita diabetes terbesar dengan kategori selain No Info didapatkan oleh mantan perokok former dengan 1590 penderita yakni 17% dari seluruh former smoker.

3. Diabetes vs hypertension

![ba-diabetes vs hypertension](https://github.com/user-attachments/assets/e9ec05ec-034c-45c2-bfe9-61f10eb9c488)


Mengecek hubungan diabetes dengan penyakit tekanan darah tinggi. Didapatkan 7.7% penderita diabetes non Hypertensi dan 38.8% penderita diabetes dan hypertensi.

4. Diabetes vs sakit jantung

![ba-diabetes vs heart disease](https://github.com/user-attachments/assets/c981bfd6-397f-41e4-9c61-83d7d3df0ada)

Pada fitur heart_disease, rerata pasien yang memiliki penyakit jantung dan menderita diabetes berjumlah 1267 pasien, yakni 38% dari pasien penderita penyakit Jantung.


**Dengan mengamati rata-rata harga relatif terhadap fitur categorical di atas, didapatkan insight sebagai berikut:**

1. Diabetes [0] menunjukkan bahwa pasien Normal Diabetes [1] menunjukkan bahwa pasien memiliki penyakit jantung

2. Pada fitur gender,didapatkan bahwa penyakit menyerang lebih banyak pada psien wanita. Grafik menunjukkan 4035 penderita diabetes laki- laki dan 4447 penderita diabetes wanita. Namun dari segi persentase,diaberes diderita 11.2% dari total pasien laki-laki dan 8.6% dari total pasien wanita.

3. Pada fitur Smoking History, penderita diabetes terbesar dengan kategori selain No Info didapatkan oleh mantan perokok former dengan 1590 penderita yakni 17% dari seluruh former smoker.

4. Pada fitur Hypertension, 2086 pasien menderita hypertension dan diabetes, hal ini memiliki pesentase yg tinggi yakni 27.8% dibandingkan dengan pasien penderita hypertension yang tidak memiliki diabetes.

5. Pada fitur heart_disease, rerata pasien yang memiliki penyakit jantung dan menderita diabetes berjumlah 1267 pasien, yakni 38% dari pasien penderita penyakit Jantung.

   
#### *Univariate Analysis - Numerical Feature*
Melihat plot histogram dan box plot fitur-fitu numerikal seperti `Age`, `BMI`, `HbA1c level`, `Blood Glucose Level`. 

![ua-numerical](https://github.com/user-attachments/assets/c85171f7-9f22-4755-97c6-f967e45ec343)


Disajikan pula dalam bentuk pair plot fitur-fitur numerik.

![Pair Plot of Numeric Features by Diabetes Classification](https://github.com/user-attachments/assets/e6000f42-4934-4ef0-b943-b87019ad7c6d)


#### *Bivariate Analysis - Numerical Feature*
1. Diabetes vs BMI
Didapatkan bahwa rerata dan median data BMI penderita diabetes lebih tinggi dari pada bukan penderita diabetes.

![ba-diabetes vs bmi](https://github.com/user-attachments/assets/3b14e4cd-c4d0-42a0-85c2-aeee617fe957)

2. Diabetes vs level HbA1c
Didapatkan bahwa rerata dan median HbA1c penderita diabetes lebih tinggi dari pada bukan penderita diabetes.

![ba-diabetes vs HbA1c](https://github.com/user-attachments/assets/ba5d6e94-6d2e-4fd4-80ab-2b59fc0aaacd)


3. Diabetes vs age
Didapatkan bahwa rerata dan median HbA1c penderita diabetes lebih tinggi dari pada bukan penderita diabetes. Didapatkan rata-rata penderita diabetes lebih tinggi dari bukan penderita, yakni 60 tahun, sedangkan bukan penderita diabetes adalah 40 tahun.

![ba-diabetes vs age](https://github.com/user-attachments/assets/187c040e-9864-4a9e-bafb-fdcacf0fdaa7)


4. Diabetes vs Blood Glucose Level
Didapatkan bahwa rerata dan median level glukosa dalam darah penderita diabetes lebih tinggi dari pada bukan penderita diabetes yakni 194, sedangkan bukan penderita diabetes adalah 132.

![ba-diabetes vs blood glucose](https://github.com/user-attachments/assets/b34f8dca-3c8c-483a-ad06-b6bf15b04b4d)


#### *Multivariate Analysis - Categorical and Numerical Feature*

1. Hubungan fitur gender, BMI dan diabetes

Di dapatkan pederita diabetes cenderung merupakan Wanita yang memiliki nilai BMI yang tinggi.


![ma-diabetes vs bmi (1)](https://github.com/user-attachments/assets/1b902685-f40f-42b6-a7d6-b5a043dd7a51)


Di dapatkan pederita diabetes cenderung merupakan Wanita yang memiliki nilai BMI yang tinggi.

2. Hubungan fitur gender, age dan diabetes

![ma-diabetes vs age vs gender](https://github.com/user-attachments/assets/08790b59-1361-422d-a1fc-1136de0e1a99)

Didapatkan penderita diabetes cenderung merupakan Wanita yang rata-rata berusia 60 tahun.

Perlu diketahui fitur mana saja yang memiliki hubungan paling kuat dengan penyakit diabetes. Hal ini dapat diketahui dengan menggunaka fungsi `corr`.

![Correlation Matrix untuk Fitur Numerical](https://github.com/user-attachments/assets/a469a827-f410-4b07-b30d-a992fb38abd1)


Berdasarkan data awal (sebelum dilakukannya training model), fitur-fitur dengan korelasi positif terhadap diabetes adalah yang memiliki nilai positif paling besar. Artinya fitur tersebut lebih mungkin memberikan kontribusi signifikan dalam membedakan kasus diabetes dan non-diabetes. Dalam hal ini, fitur yang paling berpotensi berguna adalah:

1. blood_glucose_level (0.424)

Memiliki korelasi terkuat dengan diabetes. Tingkat glukosa darah berkaitan langsung dengan diabetes, sehingga fitur ini kemungkinan menjadi prediktor yang kuat.

2. HbA1c_level (0.406)

Memiliki korelasi moderat dengan diabetes dan merupakan ukuran standar dalam mendiagnosis diabetes, menjadikannya sangat relevan.

3. Age (0.265)

Meskipun korelasinya lebih lemah, usia sering menjadi faktor risiko diabetes, sehingga mungkin dapat meningkatkan kemampuan model untuk memprediksi diabetes.

4. BMI (0.215)

Korelasi positif BMI menunjukkan bahwa fitur ini dapat membantu membedakan individu dengan dan tanpa diabetes, mengingat obesitas sering dikaitkan dengan risiko diabetes.

5. Hypertension (0.196) dan Heart Disease (0.171)

Memiliki korelasi terlemah di antara fitur yang tercantum, namun tetap dapat berperan. Hipertensi dan penyakit jantung sering dikaitkan dengan diabetes, meskipun mungkin bukan sebagai prediktor utama.


## Data Preparation

_Data preparation_ yang digunakan di antaranya:

1. Dropping _duplicated rows_. Diketahu data memiliki 3,854 _duplicate rows_ dalam 9 kolom data. Untuk itu diperlukan pembersihan data dari _duplicated rows_ dengan menggunakan :

	
		duplicate_rows_data = diabetic_disease[diabetic_disease.duplicated()]
		print("number of duplicate rows: ", duplicate_rows_data.shape)

		diabetic_disease = diabetic_disease.drop_duplicates()

		a = diabetic_disease[diabetic_disease.duplicated()].value_counts()
		print("number of duplicate rows after: ", a)

2. Seleksi Data: Menyeleksi data apakah data tersebut ada yang kosong atau tidak, jika ada data kosong maka akan dihapus.
`isnull().sum()` merupakan command yang digunakan untuk mengecek apakah terdapat data yang kosong atau missing data dan menjumlahkan banyak datanya. Namun, tidak terdapat data kosong atau missing data.

3. Menghapus nilai tertentu
Terdapat jumlah _distinct value_ yang harus dibersihkan yaitu pada kolom gender. Seharusnya kolom gender memiliki 2 _distinct value_. Namun pada tabel menunjukkan fitur memiliki 3 _distinct value_. Karena jumlah kategori 'other' jauh lebih sedikit, maka nilai 'other' akan di bersihkan dengan :

		diabetic_disease['gender'].value_counts()  
		diabetic_disease = diabetic_disease.drop(diabetic_disease[diabetic_disease['gender'] == 'Other'].index)
		diabetic_disease['gender'].value_counts()

sehingga menghasilkan : 

| Gender | Count  |
|--------|--------|
| Female | 56161  |
| Male   | 39967  |

4. Klasifikasi fitur
Fitur-fitur pada data dapat diklasifikasikan menjadi 2 kategori; yaitu fitur kategorikal dan fitur numerikal. 

		categorical_feature = ['gender', 'smoking_history', 'hypertension', 'heart_disease']
		numerical_feature = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']

5. Menangani Outlier: Melakukan pengecekan apakah data `diabetic_disease` memiliki data outlier. Dalam menangani _outlier_, digunakan metode IQR. Ditemukan _outlier_ pada data `diabetic_disease`, hal ini ditemukan dengan melakukan visualisasi dengan `boxplot`. Untuk mengidentifikasi _outlier_ yang ada, maka digunakan metode IQR. Sebagaimana kita telah lihat, seluruh fitur memiliki outlier yang cukup banyak. Mengingat data ini adalah data medis yang dapat memberikan wawasan penting mengenai kondisi langka, variasi individual pasien, dan potensi risiko kesehatan, data outlier tidak dihapuskan.


| Feature             | Count  |
|---------------------|--------|
| Age                 | 0      |
| Hypertension        | 7461   |
| Heart Disease       | 3923   |
| BMI                 | 5354   |
| HbA1c Level         | 1312   |
| Blood Glucose Level | 2031   |
| Diabetes            | 8482   |

![ba-outliers age,bmi,HbAic,bloodglucose](https://github.com/user-attachments/assets/2e864465-def0-4e86-82c0-6b5bf565e72b)


6. Melakukan re-grouping Smoking History
Data smoki1ng history yang dapat memberikan informasi bagus untuk mengetahui korelasinya dengan diabetes, memiliki terlalu banyak data `No Info`. Untuk itu dilakukan regrouping dengan menyatukan `No Info` dengan `never`, dan ketegori `ever`, `former`, dan `not curret` menjadi `past_smoker`.

| Smoking History | Count  |
|-----------------|--------|
| Non-Smoker      | 67276  |
| Past Smoker     | 19655  |
| Current         | 9197   |

7. Melakukan Label Encoder: Melakukan proses encoding terhadap `categorical_feature`. Hal ini dilakukan karena fitur-fitur kategorikal perlu dirubah agar dapat digunakan pada tahap _modeling_. Fungsi yang digunakan adalah `perform_one_hot_encoding` yang digunakan untuk mengubah kolom kategori, seperti gender dan smoking_history, menjadi kolom-kolom baru yang berisi nilai 0 atau 1

		data = perform_one_hot_encoding(data, 'gender')
		data = perform_one_hot_encoding(data, 'smoking_history')
8. Data Imbalance
Dari hasil Exploratory Data Analysis (EDA), dataset menunjukkan ketidakseimbangan (dengan 9% kasus positif diabetes dan 91% kasus negatif), sehingga penting untuk menyeimbangkan data agar model tidak bias terhadap kelas mayoritas. Untuk tujuan ini, digunakan Synthetic Minority Over-sampling Technique (SMOTE), yang menghasilkan sampel sintetis untuk kelas minoritas.

  		over = SMOTE(sampling_strategy=0.1)
		under = RandomUnderSampler(sampling_strategy=0.5)

10. Feature Selection with SelectKBest
SelectKBest digunakan untuk memilih fitur-fitur teratas yang memiliki kekuatan prediktif paling besar terhadap variabel target (diabetes). Parameter `score_func=f_classif` menunjukkan bahwa kita menggunakan nilai `F ANOVA` (sebagai ukuran signifikansi statistik) untuk mengurutkan setiap fitur, kemudian memilih 5 fitur terbaik. Hal ini membantu mengurangi dimensi data dan hanya menyimpan fitur-fitur yang paling berkorelasi dengan target.

11. Memisahkan Data menjadi Fitur dan Target
X didefinisikan sebagai semua kolom kecuali kolom diabetes, dan y ditetapkan sebagai variabel target (diasumsikan berada di kolom yang dinamai diabetes).

		X = data.drop('diabetes', axis=1)  # Assuming 'diabetes' is the target column
		y = data['diabetes']

13.  Standarisasi dan Column Transformer: membantu membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. ColumnTransformer memungkinkan kita melakukan pra-pemrosesan yang berbeda untuk fitur numerik dan fitur kategorikal: 
- Fitur Numerik: `StandardScaler()` menormalisasi fitur-fitur age, bmi, HbA1c_level, blood_glucose_level, hypertension, dan heart_disease agar memiliki rata-rata 0 dan standar deviasi 1, yang dapat meningkatkan kinerja model, terutama untuk model yang sensitif terhadap skala fitur.

- Fitur Kategorikal: `OneHotEncoder()` mengubah kolom kategorikal seperti gender dan smoking_history menjadi kolom indikator biner. handle_unknown='ignore' memastikan bahwa kategori baru atau tidak dikenal akan diabaikan.

		preprocessor = ColumnTransformer( transformers=[
        ('num', StandardScaler(), ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'hypertension', 'heart_disease']),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
         [col for col in X.columns if col.startswith('gender') or col.startswith('smoking_history')])])

13. Melakukan Splitting: membagi data menjadi _training_ dan _testing_ untuk _modeling_. Dalam melakukan _splitting_, digunakan rasio 80:20, yang berarti 80% data training, dan 20% data testing.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Proses ini dilakukan dengan menggunakan empat Algoritma. 

1. **_Models_**
   
- **_K-NN_**

	Algoritma _K-Nearest Neighbor_ (K-NN) adalah algoritma _machine learning_ yang sederhana dan mudah diterapkan, yang mana umumnya digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Algoritma ini termasuk dalam _supervised learning_. Tujuan dari algortima K-NN adalah untuk mengidentifikasi _nearest neighbor_ dari titik yang diberikan, sehingga dapat menetapkan label prediksi ke titik tersebut.

  		'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)

	- `n_neighbors=5` : Parameter ini menentukan jumlah _nearest neighbor_ yang akan dipertimbangkan saat membuat prediksi. Dalam hal ini, model akan mempertimbangkan 5 tetangga terdekat untuk mengklasifikasikan suatu titik data.


- **_Random Forest_**

  	_Random forest_ adalah kombinasi dari masing – masing _tree_ atau pohon, yang kemudian dikombinasikan ke dalam satu model. _Random Forest_ bergantung pada sebuah nilai vector acak dengan distribusi yang sama pada semua pohon yang masing masing _tree_ memiliki kedalaman yang maksimal.

		'Random Forest': RandomForestClassifier(n_estimators=50, n_jobs=-1)

   	- `n_estimators=50`: Parameter ini menentukan jumlah pohon dalam hutan. Dalam hal ini, hutan terdiri dari 50 pohon. Menggunakan jumlah pohon yang lebih sedikit dapat membuat model lebih sederhana dan lebih cepat dalam pelatihan, meskipun akurasi mungkin sedikit menurun.
	- `n_jobs=-1`: Parameter ini menentukan jumlah inti CPU yang digunakan selama pelatihan. Dengan menetapkan nilai -1, maka semua inti yang tersedia akan digunakan, yang dapat mempercepat pelatihan, terutama dengan dataset yang besar.


- **_Logistic Regression_**

  	_Logistic Regression_ adalah algoritma pembelajaran mesin **terawasi** yang digunakan untuk tugas klasifikasi biner. Algoritma ini bekerja baik ketika hubungan antara fitur dan target bersifat linier.

		'Logistic Regression': LogisticRegression(max_iter=500)

	- `max_iter=500`: Parameter ini menentukan jumlah iterasi maksimum yang dapat digunakan solver untuk menemukan koefisien logistik yang optimal. Jika solver tidak dapat mencapai konvergensi dalam 500 iterasi, maka pelatihan akan dihentikan. Meningkatkan nilai `max_iter` bisa membantu ketika data yang digunakan besar atau modelnya kompleks.


 - **_SVM_** adalah algoritma _Supervised Machine Learning_ yang digunakan untuk tugas klasifikasi dan regresi. Algoritma ini bekerja dengan mencari hyperplane optimal yang memisahkan kelas yang berbeda dalam ruang berdimensi tinggi. SVM juga dapat menggunakan kernel untuk memproyeksikan data ke dimensi yang lebih tinggi untuk klasifikasi non-linier.

		'Support Vector Machine': SVC(kernel='linear')

	- `kernel='linear'`: Parameter ini menentukan jenis kernel yang digunakan. Linear kernel berarti SVM akan menggunakan hyperplane linier untuk memisahkan kelas-kelas data. Ini lebih cepat secara komputasi dibandingkan dengan kernel non-linier seperti RBF (Radial Basis Function), dan berguna ketika data hampir bisa dipisahkan secara linier.

2. **_Langkah pemodelan_**
   
   	a. Membuat pipeline
    	Pada langkah ini, pipeline dibuat untuk mengintegrasikan proses preprocessing data, oversampling/undersampling untuk menangani data yang tidak seimbang, serta model klasifikasi yang akan digunakan.

 
   		clf = imbPipeline(steps=[('preprocessor', preprocessor),
                             ('over', over),
                             ('under', under),
                             ('classifier', model)])


   terdapat tiga langkah dasar membuat pipeline:
	- _preprocessor_: Melakukan preprocessing data.
	- _over dan under_: Menangani sampling berlebih dan kurang.
	- _classifier_: Model klasifikasi.

   	b. Pengaturan Grid Search dan _Hypermarameter Tuning_
	Untuk mengetahui model yang paling efektif bersama dengan setting _hyperparameter_ yang paling baik, digunakan teknik _hyperparameter tuning_. Hal ini dilakukan sebagai tahap persiapan untuk melakukan `GridSearchCV`. `GridSearchCV` digunakan untuk mencoba setiap kombinasi nilai _hyperparameter_ yang ada dalam grid dan mengevaluasi kinerja model untuk setiap kombinasi tersebut.

    	grid_search = GridSearchCV(clf, param_grids[model_name], cv=5, scoring='roc_auc')
   	

	c. Melatih Grid Search
	Data pelatihan (X_train, y_train) digunakan untuk melatih model sekaligus menyesuaikan _hyperparameter_.

		grid_search.fit(X_train, y_train)	


	d. Memprediksi dengan model
    	Model yang telah dilatih digunakan untuk memprediksi hasil pada data uji (X_test).

	 	y_test_pred = grid_search.predict(X_test)

Didapatkan hasil Hyperparameter tiap model adalah: 

| Model                  | Hyperparameter Name          | Value              |
|------------------------|------------------------------|--------------------|
| K-Nearest Neighbors    | classifier__n_neighbors      | 7                  |
|                        | classifier__weights          | uniform            |
| Logistic Regression    | classifier__C               | 10                 |
|                        | classifier__penalty          | l2                 |
| Support Vector Machine | classifier__C               | 10                 |
|                        | classifier__kernel           | rbf                |
| Random Forest          | classifier__max_depth        | 20                 |
|                        | classifier__min_samples_leaf | 4                  |
|                        | classifier__min_samples_split| 10                 |
|                        | classifier__n_estimators     | 200                |

- **K-Nearest Neighbors (KNN)**:
	- `classifier__n_neighbors`: 7
	Ini menunjukkan bahwa model KNN menggunakan 7 tetangga terdekat untuk melakukan klasifikasi. Artinya, ketika memprediksi kelas sebuah data baru, model akan melihat kelas mayoritas dari 7 tetangga terdekat.

	- `classifier__weights`: 'uniform'
	Mengindikasikan bahwa setiap tetangga memiliki bobot yang sama (tidak ada perbedaan berdasarkan jarak). Jadi, kontribusi semua tetangga dihitung secara merata.

- **Logistic Regression** :
	- `classifier__C`: 10
	Parameter C mengontrol regularisasi model. Nilai yang lebih tinggi (10 dalam kasus ini) mengurangi regularisasi, sehingga model lebih kompleks dan mungkin lebih cocok untuk dataset dengan pola yang lebih rumit.

	- `classifier__penalty`: 'l2'
	Ini menunjukkan bahwa model menggunakan regularisasi L2, yang berfungsi untuk menghindari overfitting dengan menambahkan penalti berdasarkan kuadrat bobot parameter.

- **Support Vector Machine (SVM)**:
	- `classifier__C`: 10
	Parameter C mengontrol tingkat regularisasi. Dengan C=10, model lebih fokus pada memaksimalkan margin antar kelas tetapi tetap memperhatikan kesalahan klasifikasi untuk mendukung data yang lebih kompleks.

	- classifier__kernel: 'rbf'
	Kernel RBF (Radial Basis Function) adalah fungsi non-linear yang cocok untuk dataset dengan pola non-linear. Kernel ini memproyeksikan data ke dimensi yang lebih tinggi untuk membuat data lebih mudah dipisahkan.

- **Random Forest**: 
	- `max_depth`: 20:
   	Ini menunjukkan bahwa kedalaman maksimum trees in the forest adalah 20 level. Membatasi tree's depth membantu mengurangi overfitting.
	- `min_samples_leaf` : 4
   	Ini berarti bahwa setiap daun (node akhir dari decision tree, tempat prediksi dilakukan) harus berisi setidaknya empat sampel.
	- `min_samples_split` : 10
 	Ini menunjukkan bahwa sebuah node harus berisi setidaknya sepuluh sampel agar dapat dibagi (untuk membuat sepuluh child node).
	- `n_estimators` : 200
  	Ini adalah jumlah decision trees. Algoritma Random Forest bekerja dengan merata-rata prediksi dari banyak decision trees untuk menghasilkan prediksi akhir, yang membantu mengurangi overfitting dan variansi.





## Evaluation
Pada pelatihan ini akan dilakukan perhitungan evaluasi dengan menggunakan beberapa metrik.

**Metrik Evaluasi**

- _Confusion Matrix_
	Metrik evaluasi yang digunakan pada project ini adalah `Confusion Matrix`. . Confusion matrix digunakan untuk memvisualisasikan kinerja model. 

<img width="450" alt="eval-confusion" src="https://github.com/user-attachments/assets/9910a852-8343-4a1e-ab7f-0435e3ebfadd">

Matriks ini menunjukkan jumlah prediksi _true positive_, _true negative_, _false positive_, dan _false negative_ yang dihasilkan oleh model. Dengan diketahuinya _true positive_, _true negative_, _false positive_, dan _false negative_ ; parameter seperti  _Precision_, _Recall_ dan _F1-Score_ dapat dihitung. 

- _Precision_
  Presisi adalah ukuran yang menunjukkan seberapa banyak prediksi true positive yang benar-benar sesuai. Presisi didefinisikan sebagai jumlah true positive (TP) dibagi dengan jumlah total true positive (TP) dan false positive (FP).

	<img width="156" alt="Precision" src="https://github.com/user-attachments/assets/a7ece7cc-8a29-40ac-a377-653b0e0de463">

- _Recalls_
  Recall (atau Sensitivitas) adalah ukuran yang menunjukkan seberapa banyak kasus positif yang berhasil diidentifikasi dengan benar. Recall didefinisikan sebagai jumlah _true positive_ (TP) dibagi dengan jumlah total _true positive_ (TP) dan _false negative_ (FN).
  
	<img width="127" alt="Recall" src="https://github.com/user-attachments/assets/af2e1730-d276-44e2-8385-b585b5237aa6">

- _F1 Score_
  F1-Score adalah rata-rata harmonik antara Precision dan Recall yang bertujuan untuk menemukan keseimbangan antara keduanya. F1-Score didefinisikan sebagai 2 kali hasil perkalian Precision dan Recall, dibagi dengan jumlah Precision dan Recall.

 	 <img width="202" alt="F1 score" src="https://github.com/user-attachments/assets/0aacc372-01f7-4a78-b83e-e81d4e379260">

- _Support_
  Support menunjukkan jumlah data  kategori tersebut.

- _Akurasi_ menunjukkan persentase prediksi yang benar (untuk semua kelas) dari total prediksi yang dibuat.
  Accuracy adalah proporsi prediksi yang benar dari total keseluruhan data. Rumusnya:

	<img width="437" alt="Accuracy" src="https://github.com/user-attachments/assets/dd07b6e7-836d-455c-826b-64f516c31336">

- _Macro average_
  _Macro average_ adalah metode perhitungan rata-rata metrik (seperti precision, recall, atau F1-score) di seluruh kelas, di mana setiap kelas memiliki bobot yang sama, tanpa memperhitungkan jumlah data (support) dalam setiap kelas.

	<img width="424" alt="macro_average" src="https://github.com/user-attachments/assets/63c0d386-b080-4738-a229-3a575f2d1f66">

- _Weight average_
  _Weighted average_ adalah metode perhitungan rata-rata metrik (seperti precision, recall, atau F1-score) di seluruh kelas dengan memberikan bobot sesuai dengan jumlah data (support) dalam setiap kelas.

	<img width="533" alt="weighted rata-rata berbobot" src="https://github.com/user-attachments/assets/5ea3bf0c-70d6-47ab-9dfb-35284d644197">

**Perbandingan Metrik**

| **Metrik**     | **Sisi Positif**                                    | **Sisi Negatif**                               | **Kecocokan Penggunaan**                                                     |
|----------------|-----------------------------------------------------|------------------------------------------------|-------------------------------------------------------------------------------|
| **Precision**  | Mengurangi false positive                           | Tidak memperhatikan false negative             | Saat false positive berisiko tinggi .             |
| **Recall**     | Mengurangi false negative                           | Tidak memperhatikan false positive             | Saat false negative berisiko tinggi .          |
| **F1-Score**   | Menyeimbangkan precision dan recall                 | Tidak memperhatikan true negative              | Saat dataset tidak seimbang dan precision/recall sama pentingnya.            |
| **Accuracy**   | Mudah dihitung dan diinterpretasikan                | Menyesatkan untuk dataset tidak seimbang       | Saat dataset seimbang dan semua kelas sama pentingnya.                        |
| **Support**    | Memberikan informasi distribusi kelas               | Bukan metrik kinerja                           | Untuk memahami distribusi kelas dalam dataset.                                |

Mengingat kelebihan dan kekurangan parameter _Precision_, _Recall_, _Accuracy F1-Score_, diketahui bahwa evaluasi dengan F1-Score lebih cocok digunakan pada pembahasan ini karena F1-Score cocok untuk data imbalance seperti data yang kita gunakan. 

F1-score sangat cocok untuk digunakan pada data yang tidak seimbang (imbalanced data) karena F1-score memperhitungkan kedua metrik penting, yaitu _precision_ dan _recall_, yang memberikan gambaran lebih lengkap tentang kinerja model, terutama ketika distribusi kelas tidak merata. _Precision_ mengukur seberapa banyak prediksi positif yang benar, sementara recall mengukur seberapa banyak prediksi positif yang berhasil ditemukan dari total data positif yang ada.


**Langkah-langkah diambil untuk melakukan evaluasi adalah:**
1. Memprediksi Data Uji

   		 y_test_pred = grid_search.predict(X_test)
   
Model yang telah dilatih digunakan untuk memprediksi label pada data uji (X_test).

2. Menghitung Metrik Evaluasi (_Precision_, _Recall_, _F1-Score_, _Accuracy_)

		precision = precision_score(y_test, y_test_pred, average='weighted')
		recall = recall_score(y_test, y_test_pred, average='weighted')
		f1 = f1_score(y_test, y_test_pred, average='weighted')
		accuracy = accuracy_score(y_test, y_test_pred)
		support = len(y_test)

   Metrik seperti _precision_, _recall_, _F1-score_, _accuracy_, dan jumlah data uji (_support_) dihitung untuk menilai performa model. 

3. Mencetak Laporan Klasifikasi
	
 		print(classification_report(y_test, y_test_pred, target_names=['Class 0', 'Class 1']))

   Laporan klasifikasi mencakup metrik seperti precision, recall, F1-score, dan support untuk setiap kelas.
   
4. Menghitung dan memvisualisasi `Confusion Matrix`

		cm = confusion_matrix(y_test, y_test_pred)
		sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])


_Confusion Matrix_ dihitung untuk menunjukkan distribusi prediksi model terhadap label sebenarnya.
     
5. Menyimpan Hasil Evaluasi

		metrics_results['Model'].append(model_name)
		metrics_results['Precision'].append(precision)
		metrics_results['Recall'].append(recall)
		metrics_results['F1-Score'].append(f1)
		metrics_results['Accuracy'].append(accuracy)
		metrics_results['Support'].append(support)

   Hasil evaluasi setiap model disimpan dalam bentuk struktur data untuk memudahkan analisis dan perbandingan.
   
6. Menyortir dan Menampilkan Hasil Evaluasi

   		sorted_metrics_df = metrics_df.sort_values(by='F1-Score', ascending=False)
		print(sorted_metrics_df)

    Hasil evaluasi disusun berdasarkan metrik utama (F1-score) untuk mengidentifikasi model terbaik.
   
11. Memvisualisasikan Perbandingan Metrik

    	metrics_df.set_index('Model')[['Precision', 'Recall', 'F1-Score', 'Accuracy']].plot(kind='bar', figsize=(12, 6))

	Visualisasi performa model berdasarkan metrik evaluasi dalam diagram batang.

**Hasil Evaluasi**

1. **K-Nearest Neighbors (KNN)**:
Hasil _Confusion Matrix_ untuk model KKN adalah :

![KKN-confusion](https://github.com/user-attachments/assets/fe5f6caa-05cc-4300-ad78-15fc5922be14)

Hasil _Classification Report_ untuk Model KKN adalah : 

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| **Class 0** | 0.98      | 0.93   | 0.95     | 17525   |
| **Class 1** | 0.53      | 0.80   | 0.64     | 1701    |
| **Accuracy**|           |        | 0.92     | 19226   |
| **Macro Avg**| 0.75      | 0.86   | 0.80     | 19226   |
| **Weighted Avg**| 0.94      | 0.92   | 0.93     | 19226   |

Laporan klasifikasi untuk model K-Nearest Neighbors menunjukkan kinerja yang sangat baik pada Kelas 0, dengan precision tinggi (0.98), recall (0.93), dan F1-score (0.95), yang mencerminkan prediksi yang akurat dan andal untuk kelas mayoritas.

Namun, model menghadapi kesulitan dalam memprediksi Kelas 1, dengan precision yang lebih rendah (0.53) dan F1-score (0.64), meskipun recall-nya cukup tinggi (0.80), yang menunjukkan bahwa sebagian besar instance Kelas 1 yang sebenarnya berhasil terdeteksi. 

Akurasi keseluruhan sebesar 92% mencerminkan kinerja umum yang baik, tetapi perbedaan kinerja antar kelas mengindikasikan pengaruh ketidakseimbangan data, di mana Kelas 0, dengan 17.525 instance, jauh lebih dominan dibandingkan Kelas 1 yang hanya memiliki 1.701 instance.

2. **_Random Forest_**:
Hasil _Confusion Matrix_ untuk model Random Forest adalah :

![RandomForest-CM](https://github.com/user-attachments/assets/6b2b4ac6-7df0-4772-8ff4-dee8bbe82e30)


Hasil _Classification Report_ untuk Model Random Forest adalah : 

| Metric         | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Class 0        | 0.98      | 0.96   | 0.97     | 17525   |
| Class 1        | 0.64      | 0.81   | 0.72     | 1701    |
| Accuracy       | -         | -      | 0.94     | 19226   |
| Macro Avg      | 0.81      | 0.88   | 0.84     | 19226   |
| Weighted Avg   | 0.95      | 0.94   | 0.95     | 19226   |

Model ini menunjukkan kinerja sangat baik untuk Class 0 dengan precision (0.98), recall (0.96), dan F1-score (0.97), namun kurang optimal untuk Class 1, dengan precision (0.64) dan F1-score (0.72) yang lebih rendah meskipun recall tinggi (0.81). Akurasi keseluruhan mencapai 94%, namun rata-rata makro (precision 0.81, recall 0.88, F1-score 0.84) menunjukkan ketidakseimbangan kelas. Rata-rata berbobot yang lebih tinggi (precision 0.95, recall 0.94, F1-score 0.95) mencerminkan dominasi Class 0. 

3. **_Logistic Regression_**:
Hasil _Confusion Matrix_ untuk model Logistic Regression adalah :

![RandomForest-CM](https://github.com/user-attachments/assets/6b2b4ac6-7df0-4772-8ff4-dee8bbe82e30)


Hasil _Classification Report_ untuk _Logistic Regression_ adalah : 

| Metric         | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Class 0        | 0.98      | 0.94   | 0.96     | 17525   |
| Class 1        | 0.55      | 0.79   | 0.65     | 1701    |
| Accuracy       | -         | -      | 0.92     | 19226   |
| Macro Avg      | 0.76      | 0.86   | 0.80     | 19226   |
| Weighted Avg   | 0.94      | 0.92   | 0.93     | 19226   |


Model ini menunjukkan kinerja yang sangat baik untuk Class 0 dengan precision (0.98) dan F1-score (0.96), tetapi kinerjanya kurang optimal untuk Class 1 dengan precision (0.55) dan F1-score (0.65), meskipun recall cukup tinggi (0.79). Akurasi keseluruhan model adalah 92%, dan rata-rata makro (precision 0.76, recall 0.86, F1-score 0.80) menunjukkan ketidakseimbangan antara kedua kelas. Rata-rata berbobot (precision 0.94, recall 0.92, F1-score 0.93) mencerminkan kinerja yang lebih baik pada Class 0. Hal ini menunjukkan bahwa model perlu disesuaikan lebih lanjut untuk meningkatkan prediksi pada kelas minoritas (Class 1).

4. **_Support Vector Machine_**:
Hasil _Confusion Matrix_ untuk model _Support Vector Machine_ adalah :

![SVM-CF](https://github.com/user-attachments/assets/0ac3ef04-d2bf-4cc5-84d0-f43d53a5f98e)


Hasil _Classification Report_ untuk _Support Vector Machine_ adalah : 

| Metric         | Precision | Recall | F1-Score | Support |
|----------------|-----------|--------|----------|---------|
| Class 0        | 0.98      | 0.95   | 0.97     | 17525   |
| Class 1        | 0.62      | 0.80   | 0.70     | 1701    |
| Accuracy       | -         | -      | 0.94     | 19226   |
| Macro Avg      | 0.80      | 0.88   | 0.83     | 19226   |
| Weighted Avg   | 0.95      | 0.94   | 0.94     | 19226   |

Model ini menunjukkan kinerja yang sangat baik untuk Class 0 dengan precision (0.98) dan F1-score (0.97), serta recall yang tinggi (0.95). Namun, untuk Class 1, precision (0.62) dan F1-score (0.70) masih dapat ditingkatkan meskipun recall (0.80) cukup baik. Akurasi keseluruhan model mencapai 94%, dengan rata-rata makro (precision 0.80, recall 0.88, F1-score 0.83) menunjukkan keseimbangan yang lebih baik antara kedua kelas. Rata-rata berbobot (precision 0.95, recall 0.94, F1-score 0.94) menyoroti dominasi kinerja model pada Class 0, yang lebih besar.

**Perbandingan Hasil Perhitungan**

| Rank | Model                  | Precision   | Recall     | F1-Score   | Accuracy   | Support |
|------|------------------------|-------------|------------|------------|------------|---------|
| 1    | Random Forest          | 0.951234    | 0.943358   | 0.946281   | 0.943358   | 19226   |
| 2    | Support Vector Machine | 0.948704    | 0.939717   | 0.943055   | 0.939717   | 19226   |
| 3    | Logistic Regression    | 0.940615    | 0.923177   | 0.929423   | 0.923177   | 19226   |
| 4    | K-Nearest Neighbors    | 0.939555    | 0.919120   | 0.926333   | 0.919120   | 19226   |

Jika dibandingkan dengan grafik, maka: 

![rekap-CF](https://github.com/user-attachments/assets/92e393e3-21dd-46ca-98be-e7add468297d)


3. Dibutuhkan juga informasi `Feature Importance` baru yang didapatkan setelah perhitungan dengan model.
   
![Feature Importance](https://github.com/user-attachments/assets/17ae83b2-557e-498f-befa-f2e633d6bb2e)

* HbA1c_level adalah fitur yang paling penting dengan nilai penting sebesar 0,408. HbA1c adalah ukuran rata-rata kadar glukosa darah selama 2 hingga 3 bulan terakhir, sehingga tidak mengherankan jika ini merupakan prediktor signifikan untuk diabetes.

* Blood_glucose_level adalah fitur kedua yang paling penting dengan nilai penting sebesar 0,318. Hal ini sejalan dengan pengetahuan medis, karena kadar glukosa darah langsung digunakan untuk mendiagnosis diabetes.

* Age adalah fitur ketiga yang paling penting dengan nilai penting sebesar 0,134. Sudah diketahui bahwa risiko diabetes tipe 2 meningkat seiring bertambahnya usia.

* BMI menduduki peringkat keempat dalam hal pentingnya, yaitu sebesar 0,08. Indeks Massa Tubuh (BMI) adalah faktor risiko utama untuk diabetes, dan peranannya telah didokumentasikan dengan baik dalam literatur medis.


## Conclusion
Dari evaluasi, **Random Forest** memiliki nilai F1 Score yang cukup tinggi yaitu 0.946281.

Dari tabel di atas, model Random Forest menunjukkan kinerja terbaik dengan nilai precision (0.95), recall (0.94), dan F1-score (0.95), diikuti oleh Support Vector Machine dengan skor yang sangat mendekati. Logistic Regression dan K-Nearest Neighbors berada di posisi ketiga dan keempat, masing-masing dengan performa yang sedikit lebih rendah namun tetap menunjukkan hasil yang solid. Secara keseluruhan, semua model memiliki accuracy di atas 92%, dengan Random Forest memberikan hasil terbaik dalam hal keseimbangan antara precision, recall, dan F1-score.

Analisis pentingnya fitur menyoroti bahwa HbA1c_level dan blood_glucose_level merupakan faktor paling penting dalam memprediksi diabetes. Usia (age) dan BMI juga memiliki pengaruh yang signifikan. Namun, beberapa fitur seperti riwayat merokok (smoking history) dan jenis kelamin (gender) memiliki dampak minimal atau bahkan tidak memengaruhi prediksi model.


Referensi:

[1] [A risk assessment and prediction framework for diabetes mellitus using
machine learning algorithms] (https://www.sciencedirect.com/science/article/pii/S2772442523001405)

[2] [Diabetic Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

