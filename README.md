# Histogram-Equalization-by-Parallel-Computing
　　直方圖均衡化（Histogram Equalization）是影像處理（Digital Image Processing）中利用統計影像色階值的直方圖來對影像對比度進行調整的方法，而此方法需大量統計與查表並計算出新的像素色階值，其疊代的次數與影像像素點多寡有關。而為了加快的運算時間，因此探討使用平行化框架，如：OpenMP、OpenCL 改善其效能比例為何，且分別比較使用 CPU 和 GPU 加速的差異如何。
