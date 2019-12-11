# Aplikasi Sentiment Analisis Twitter Pemilu dengan Menggunakan Metode ANN Artificial Neural Network

## Python Version

Python 3.6.2

## Library Python

- Flask                  1.0.2
- Flask-Bootstrap        3.3.7.1
- h5py                   2.9.0
- html5lib               0.9999999
- Keras                  2.2.4
- matplotlib             3.0.2
- nltk                   3.4
- numpy                  1.15.4
- pandas                 0.23.4
- scikit-learn           0.20.2
- scipy                  1.2.0
- sklearn                0.0
- tensorflow             1.5.0

## Disciption

Aplikasi ini di buat dengan konsep machine learning ada beberapa tahapan untuk membangun aplikasi ini
 1. Crowling data twitter kemudian data di bagi menjadi 70% untuk training dan 30% untuk testing.
 2. Cleaning dan stemming data training, aplikasi ini stemming menggunakan sastrawi
 3. Labeling data training dengan angaka (1) positif, (0) negatif
 4. Membuat arsitektur model dengan tensorflow
 5. Melakukan training hingga mendapatkan akurasi yang bagus dengan mengkombinasikan beberapa parameter seperti jumlah iterasi atau merubah arsitekturnya.
 6. Setelah mendapat akurasi yang bagus eksport model ke format h5 atau pkt
 7. Melakukan testing
 8. Membuat interface dan selesai

## Pengembangan

Menambahkan hidden layer atau bisa di kembangkan lagi mengunakan CNN Backpropagation agar bisa melakukan pembelajaran mesin automatis
