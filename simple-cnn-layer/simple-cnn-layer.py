import numpy as np

def conv2d(x, W, b):
    """
    Channels-first düzeninde 2D Konvolüsyon İleri Besleme (Forward Pass)
    Valid padding, stride=1.
    
    Boyutlar:
    x: (N, C_in, H, W)
    W: (C_out, C_in, KH, KW)
    b: (C_out,)
    
    Çıktı:
    y: (N, C_out, H_out, W_out)
    """
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    N, C_in, H, W_in = x.shape
    C_out, _, KH, KW = W.shape
    
    # 1. Çıktı Boyutlarını Hesapla (Valid Padding & Stride=1)
    H_out = H - KH + 1
    W_out = W_in - KW + 1
    
    # Çıktı matrisini sıfırlarla ilklendir
    y = np.zeros((N, C_out, H_out, W_out), dtype=np.float64)
    
    # 2. Döngüleri optimize etmek ve H/W iç döngülerinden kaçınmak için pencereleri tarıyoruz
    # N, C_out ve C_in üzerindeki açık döngülere izin verilmiştir.
    for n in range(N):
        for c_out in range(C_out):
            # Her bir çıktı kanalı için ilgili filtreyi ve bias değerini al
            kernel = W[c_out]  # Boyut: (C_in, KH, KW)
            bias = b[c_out]
            
            # Tüm girdi kanallarının toplamını tutacak geçici matris
            total_conv = np.zeros((H_out, W_out), dtype=np.float64)
            
            for c_in in range(C_in):
                # Girdi ve filtrenin ilgili kanallarını seç
                x_channel = x[n, c_in]  # Boyut: (H, W_in)
                k_channel = kernel[c_in] # Boyut: (KH, KW)
                
                # İç H/W döngülerini yok etmek için matris kaydırma (sliding window) matrisleşmesi:
                # Kernel boyutu kadar (KH x KW) kaydırma matrislerini üst üste ekliyoruz.
                for u in range(KH):
                    for v in range(KW):
                        # x_channel üzerinden dilimleri tek seferde (vektörize) çarpıp ekliyoruz
                        total_conv += x_channel[u : u + H_out, v : v + W_out] * k_channel[u, v]
            
            # Bias değerini ekle ve çıktı matrisine ata
            y[n, c_out] = total_conv + bias
            
    return y