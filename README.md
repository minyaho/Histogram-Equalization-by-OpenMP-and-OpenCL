# Histogram-Equalization-by-Parallel-Computing
### 前言：
　　直方圖均衡化（Histogram Equalization）是影像處理（Digital Image Processing）中利用統計影像色階值的直方圖來對影像對比度進行調整的方法，而此方法需大量統計與查表並計算出新的像素色階值，其疊代的次數與影像像素點多寡有關。而為了加快的運算時間，因此探討使用平行化框架，如：OpenMP、OpenCL 改善其效能比例為何，且分別比較使用 CPU 和 GPU 加速的差異如何。

<br>  

### 背景知識：

* 影像直方圖（Image Histogram）  

    影像直方圖（Image Histogram）是一種用來表現數位影像中像素色階值分布的直方圖，根據統計影像中不同亮度的像素總數，可畫出代表該影像的一張直方圖，透過這張直方圖可以一眼就看出此圖中像素色階值（以下簡稱像素值）的分布情形。如圖１為範例圖像，而圖２為該影像的直方圖。

    <br><div align=center><img width="450" src="https://github.com/minyaho/Histogram-Equalization-by-Parallel-Computing/blob/master/readme_images/01.png"/></div><br>

    而直方圖也能用來呈現一張影像的曝光程度，常用於影像處理或是拍攝時調整相機參數之用。如圖３的影像曝光適中，其像素值集中分佈在中間區域，代表照片曝光均勻，不偏亮偏暗。圖４則是過亮，像素值集中分佈在右半邊，代表照片出現大量偏白，即過亮的像素，反映照片偏亮，例如是貓咪站立的地面、欄柵之間的位置等等。圖５則是過暗，像素值集中分佈在左半邊，代表照片出現大量偏黑，即過暗的像素，反映照片整體偏暗。  
    
    <br><div align=center><img width="300" src="https://github.com/minyaho/Histogram-Equalization-by-Parallel-Computing/blob/master/readme_images/02.png"/></div>  <br>

    而均衡化的作法是運用累積分布函數（Cumulative Distribution Function , CDF）對像素值進行調整以實現對比度增強，把原始圖像的像素直方圖從比較集中的某個像素區間變成在全部像素範圍內的均勻分佈，如圖７所示。

    <br><div align=center><img width="300" src="https://github.com/minyaho/Histogram-Equalization-by-Parallel-Computing/blob/master/readme_images/03.png"/></div>  <br>
    
    在此討倫其轉換公式 ℎ 如下:
    
    <br><div align=center><img width="300" src="https://github.com/minyaho/Histogram-Equalization-by-Parallel-Computing/blob/master/readme_images/04.png"/></div>  <br>
    
    說明:

      1. 𝑐𝑑𝑓: 累積分布函數，其實就是機率密度函數（PDF）的積分，因此可以從直方圖中獲得各個像素值的 PDF 後再將 PDF 做累加求出 𝑐𝑑𝑓 。
      2. 𝑐𝑑𝑓_𝑚𝑖𝑛: 累積分布函數中的最小值，例如像素值 20 共出現 3 次是 𝑐𝑑𝑓 中最小的。
      3. 𝑀和𝑁: 分別代表了圖像的長寬大小
      4. 𝐿: 像素值級數，如 8 位元的灰階深度，則像素值級數共有 2^8=256 級數
      5. 𝑣: 某像素點的色階值，若為灰階影像則該數值落在 0 ~ 255 區間
      6. round( ): 代表著四捨五入的運算
    <br>
      
* OpenMP  (Open Multi-Processing)  

    OpenMP 是一套支援跨平台共享記憶體方式的多執行緒並行的編程 API，使用C, C++ 和 Fortran 語言，可以在大多數的處理器體系和作業系統中執行。<br>  
    此外 OpenMP 提供了對平行演算法的高層次的抽象描述，特別適合在多核 CPU 機器上的並行程式設計。程式設計師通過在原始碼中加入專用的 pragma 來指明自己的意圖，由此編譯器可以自動將程式進行並列化，並在必要之處加入同步互斥以及通信。當選擇忽略這些 pragma，或者編譯器不支援 OpenMP 時，程式又可退化為通常的程式（一般為串行），程式碼仍然可以正常運作，只是不能利用多執行緒來加速程式執行。

<br>  

* OpenCL  (Open Computing Language)  

    OpenCL 是一個針對異質性計算裝置（Heterogeneous Device）進行平行化運算所設計的標準 API 以及程式語言。所謂的「異質性計算裝置」，是指在同一個電腦系統中，有兩種以上架構差異很大的計算裝置，例如一般的 CPU 以及顯示晶片 GPU，或是 DSP、FPGA 或其他類型的處理器與硬體加速器所組成。<br>  
    OpenCL 的主要設計目的，是為了提供一個容易使用、且適用於各種不同裝置的平行化計算平台。因此，它提供了兩種平行化的模式，包括 task parallel 以及 data parallel。目前 GPGPU（將高效能運算的 GPU 運用在一般運算上）的應用，主要是以 data parallel 為主，而所謂的 data parallel，指的是將大量的資料都進行同樣的處理。這種形式的平行化運算需求，其實在很多應用上都可以見到。例如，影像處理的程式，經常需要對一個影像的每個 pixel 進行同樣的動作（例如 Gaussian blur）。因此，這類工作很適合 data parallel 的模式。
 
<br>  

### 實驗方法：

* 設計均衡化演算法程式碼  

    針對下列均衡化函式，
    <br><div align=center><img width="300" src="https://github.com/minyaho/Histogram-Equalization-by-Parallel-Computing/blob/master/readme_images/04.png"/></div>  <br>
    設計下方程式碼對應。

    ```C++
    sum = round(cdf_R[IObj->sR[index]] - minCdfValue_R)/(2048*2048 - minCdfValue_R)*(255);
    sum = ((sum>255) ? 255 : ((sum<0) ? 0 : sum));
    IObj->sR[index] = (unsigned)sum;
    ```

<br>  

* 分析應平行化的程式片段  

    下方的程式碼由於需疊代的次數與影像的大小相關，若圖像大小為 2048 x 2048，那麼此段程式碼共疊代 4,192,256 ‬次。且此段程式碼不具有資料相依性，也就是這段函數的輸入並沒有上一次自己的輸出。此外這些程式碼的負擔較重，需要疊代、載入、浮點運算與判斷。而且不只有 RGB 影像通道中的 R 通道要均衡化、G 與 B 通道也是。


	```c++
    for(y=1; y!=IMG_H; y++){
    index = y*IMG_W;
    for(x=1; x!=IMG_W; x++){

      sum = round(cdf_R[IObj->sR[index]] - minCdfValue_R) / (2048*2048 - minCdfValue_R) * (255);
      sum = ((sum>255) ? 255 : ((sum<0) ? 0 : sum));
      IObj->sR[index] = (unsigned)sum;

      sum = round(cdf_G[IObj->sG[index]] - minCdfValue_G) / (2048*2048 - minCdfValue_G) * (255);
      sum = ((sum>255) ? 255 : ((sum<0) ? 0 : sum));
      IObj->sG[index] = (unsigned)sum;

      sum = round(cdf_B[IObj->sB[index]] - minCdfValue_B) / (2048*2048 - minCdfValue_B) * (255);
      sum = ((sum>255) ? 255 : ((sum<0) ? 0 : sum));
      IObj->sB[index] = (unsigned)sum;
      index++;
      }
    }
	```

<br>  

 * 平化程式碼片段 - OpenMP  
 
    OpenMP 是借助 CPU 的 thread 優勢來達到平行化加速。根據應平行化的程式片段改寫成 OpenMP 可平行化運行的片段，
    
    ```c++
    #pragma omp parallel num_threads(maxThread)
    {
      #pragma omp for schedule(dynamic,20) private(y,x,index,sum)
      for(y=1; y<IMG_H; y++){
        index = y*IMG_W;
        for(x=1; x<IMG_W; x++){
          sum = round(cdf_R[IObj->sR[index]] - minCdfValue_R) / (2048*2048 - minCdfValue_R) * (255);
          sum = ((sum>255) ? 255 : ((sum<0) ? 0 : sum));
          IObj->sR[index] = (unsigned)sum;

          sum = round(cdf_G[IObj->sG[index]] - minCdfValue_G) / (2048*2048 - minCdfValue_G) * (255);
          sum = ((sum>255) ? 255 : ((sum<0) ? 0 : sum));
          IObj->sG[index] = (unsigned)sum;

          sum = round(cdf_B[IObj->sB[index]] - minCdfValue_B) / (2048*2048 - minCdfValue_B) * (255);
          sum = ((sum>255) ? 255 : ((sum<0) ? 0 : sum));
          IObj->sB[index] = (unsigned)sum;
          index++;
        }
      }
    }
    ```

    上述的程式碼相較應平行化的程式碼多出 OpenMP 語法的部分，分別是
    
    ```c++
    #pragma omp parallel num_threads(maxThread) { }
    ```
    
    其宣告括號內的 { } 程式碼片段，以 maxThread 的數量進行平行化運算。而 maxThread 為 CPU 的硬體 Thread 量，其為可調動之參數，不同的 CPU 有不同的硬體 thread 數量，可視資源量自行調動，但最高不能超過硬體上的 thread 量。<br>  
    
    此外還有

    ```c++
    #pragma omp for schedule(dynamic,20) private(y,x,index,sum)
    ```

    其宣告括號內的 { } 程式碼片段，將會反覆地進行疊代，而每個 thread 將會份配到 20 份工作量（一次疊代視為一次工作量），而分配工作量的順序將會視 thread 的完成速度動態調整。此外每個 thread 將獲得專屬的 private 變數。<br>  
    
    本次 OpenCL 採用 2 維的全域工作大小 global_work_size，讓每份工作（ＷorkItem）以類二維矩陣的模式排列，使其發揮 GPU 的運算潛能，而這 2 維分別對應著影像的 column 與 row 。此外將每份工作所需的資料預先載入 GPU 的內存中，以便運算時快速使用，不須要再從主記憶體讀取，更能加快運算的速度。<br>  

### 實驗結果：

* 測試環境
	* 作業系統： Windows 10, 版本 1809
	* CPU： Intel Core i7 7700HQ 2.80GHz
	* GPU：NVIDIA GTX 1060 版本 CUDA 10

* 測試方法
	* 共分為單線程CPU、OpenMP、OpenCL，這 3 個測試項目
	* 每個測試項目共測試 2 張圖片
	* 每張照片各別測試 5 次並取平均
	* 因此每個測項共有 10 次的測試
	
* 測試資料

	|圖片A 風景照|圖片B 貓咪照|
	| - | - |
	|<img width="200" src="https://github.com/minyaho/Histogram-Equalization-by-Parallel-Computing/blob/master/readme_images/05.png"/>|<img width="200" src="https://github.com/minyaho/Histogram-Equalization-by-Parallel-Computing/blob/master/readme_images/06.png"/>|


