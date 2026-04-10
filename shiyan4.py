import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift

# 辅助函数 
def compute_spectrum(img):
    """计算图像频谱（中心化，对数幅度）"""
    f = fft2(img.astype(np.float32))
    fshift = fftshift(f)
    mag = np.log1p(np.abs(fshift))
    return (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

def downsample(img, factor, sigma=None):
    """下采样图像，可选高斯预滤波"""
    if sigma is not None:
        ksize = int(6 * sigma) | 1
        img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    h, w = img.shape[:2]
    return cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_NEAREST)

def gradient_based_m(img, base_factor=4):
    """根据梯度估计局部下采样因子M（梯度大则M小，梯度小则M大）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag_norm = mag / (mag.max() + 1e-8)
    M_map = base_factor - (mag_norm * (base_factor - 2))
    return M_map.astype(np.float32)

#  生成测试图 
def generate_chessboard(size=256, block_size=8):
    """生成棋盘格"""
    chess = np.zeros((size, size), np.uint8)
    for i in range(0, size, block_size):
        for j in range(0, size, block_size):
            if (i // block_size + j // block_size) % 2 == 0:
                chess[i:i+block_size, j:j+block_size] = 255
    return chess

def generate_chirp(size=256):
    """生成Chirp信号（水平方向频率递增）"""
    chirp = np.zeros((size, size), np.uint8)
    for i in range(size):
        for j in range(size):
            # 频率从左到右线性增加 0 到 0.5
            freq = j / size * 0.5
            value = 127 + 127 * np.sin(2 * np.pi * freq * j)
            chirp[i, j] = np.clip(value, 0, 255).astype(np.uint8)
    return chirp

#第一部分：混叠分析 
def part1():
    """使用棋盘格和chirp图展示混叠现象"""
    chess = generate_chessboard(size=256, block_size=8)
    chirp = generate_chirp(size=256)
    factor = 4   
    # 棋盘格处理
    chess_small_direct = downsample(chess, factor, sigma=None)
    chess_rec_direct = cv2.resize(chess_small_direct, (chess.shape[1], chess.shape[0]), interpolation=cv2.INTER_LINEAR)
    chess_small_blur = downsample(chess, factor, sigma=1.0)
    chess_rec_blur = cv2.resize(chess_small_blur, (chess.shape[1], chess.shape[0]), interpolation=cv2.INTER_LINEAR)   
    # Chirp处理
    chirp_small_direct = downsample(chirp, factor, sigma=None)
    chirp_rec_direct = cv2.resize(chirp_small_direct, (chirp.shape[1], chirp.shape[0]), interpolation=cv2.INTER_LINEAR)
    chirp_small_blur = downsample(chirp, factor, sigma=1.0)
    chirp_rec_blur = cv2.resize(chirp_small_blur, (chirp.shape[1], chirp.shape[0]), interpolation=cv2.INTER_LINEAR)   
   # 显示棋盘格结果
    titles = ['Original', 'Direct Downsample', 'Gaussian + Downsample']
    imgs = [chess, chess_rec_direct, chess_rec_blur]
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(imgs[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.suptitle('Chessboard - Spatial Domain')
    plt.show()   
    # 棋盘格频谱图
    spec_orig = compute_spectrum(chess)
    spec_direct = compute_spectrum(chess_rec_direct)
    spec_blur = compute_spectrum(chess_rec_blur)
    
    plt.figure(figsize=(12, 4))
    for i, (spec, title) in enumerate(zip([spec_orig, spec_direct, spec_blur],
                                          ['FFT Spectrum (Original)', 'FFT Spectrum (Direct Downsample)', 'FFT Spectrum (Gaussian + Downsample)'])):
        plt.subplot(1, 3, i+1)
        plt.imshow(spec, cmap='hot')
        plt.title(title)
        plt.axis('off')
    plt.suptitle('Chessboard - FFT Spectrum')
    plt.show()   
    # 显示Chirp结果
    imgs = [chirp, chirp_rec_direct, chirp_rec_blur]
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(imgs[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.suptitle('Chirp Signal - Spatial Domain (Frequency increases from left to right)')
    plt.show()   
    # Chirp频谱图
    spec_orig = compute_spectrum(chirp)
    spec_direct = compute_spectrum(chirp_rec_direct)
    spec_blur = compute_spectrum(chirp_rec_blur)
    
    plt.figure(figsize=(12, 4))
    for i, (spec, title) in enumerate(zip([spec_orig, spec_direct, spec_blur],
                                          ['FFT Spectrum (Original)', 'FFT Spectrum (Direct Downsample)', 'FFT Spectrum (Gaussian + Downsample)'])):
        plt.subplot(1, 3, i+1)
        plt.imshow(spec, cmap='hot')
        plt.title(title)
        plt.axis('off')
    plt.suptitle('Chirp Signal - FFT Spectrum')
    plt.show()
# 第二部分：验证σ公式 
def part2(chess):
    """固定M=4，测试不同σ的效果"""
    M = 4
    sigmas = [0.5, 1.0, 1.8, 2.0, 4.0]
    results = []   
    plt.figure(figsize=(18, 10))
    
    for i, sigma in enumerate(sigmas):
        ksize = int(6 * sigma) | 1
        blurred = cv2.GaussianBlur(chess, (ksize, ksize), sigma)
        down = cv2.resize(blurred, (chess.shape[1] // M, chess.shape[0] // M), interpolation=cv2.INTER_NEAREST)
        up = cv2.resize(down, (chess.shape[1], chess.shape[0]), interpolation=cv2.INTER_LINEAR)
        mse = np.mean((chess.astype(np.float32) - up.astype(np.float32))**2)
        results.append((sigma, mse, up))
        
        plt.subplot(2, len(sigmas), i+1)
        plt.imshow(up, cmap='gray')
        plt.title(f'sigma={sigma}, MSE={mse:.1f}')
        plt.axis('off')
        
        spec = compute_spectrum(up)
        plt.subplot(2, len(sigmas), i+len(sigmas)+1)
        plt.imshow(spec, cmap='hot')
        plt.title(f'FFT Spectrum sigma={sigma}')
        plt.axis('off')
    
    plt.suptitle(f'Part 2: Different sigma Values (M={M})')
    plt.tight_layout()
    plt.show()
    
    best_sigma = min(results, key=lambda x: x[1])[0]
    theoretical = 0.45 * M
    
    print(f"\nBest sigma = {best_sigma:.2f}")
    print(f"Theoretical value = 0.45*{M} = {theoretical:.2f}")
# 第三部分：自适应下采样 
def part3(chess):
    """自适应下采样：根据梯度估计局部M和σ"""
    M_uniform = 4
    sigma_uniform = 1.8   
    # 统一σ方法
    blurred_uniform = cv2.GaussianBlur(chess, (int(6*sigma_uniform)|1, int(6*sigma_uniform)|1), sigma_uniform)
    down_uniform = cv2.resize(blurred_uniform, (chess.shape[1] // M_uniform, chess.shape[0] // M_uniform), cv2.INTER_NEAREST)
    up_uniform = cv2.resize(down_uniform, (chess.shape[1], chess.shape[0]), cv2.INTER_LINEAR)   
    # 自适应方法
    M_map = gradient_based_m(chess, base_factor=4)
    sigma_map = 0.45 * M_map   
    h, w = chess.shape
    filtered = np.zeros_like(chess, dtype=np.float32)
    block_size = 32
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = chess[i:min(i+block_size, h), j:min(j+block_size, w)]
            sigma_block = np.mean(sigma_map[i:min(i+block_size, h), j:min(j+block_size, w)])
            if sigma_block > 0.2:
                ksize = int(6 * sigma_block) | 1
                if ksize < 3:
                    ksize = 3
                blurred_block = cv2.GaussianBlur(block, (ksize, ksize), sigma_block)
                filtered[i:min(i+block_size, h), j:min(j+block_size, w)] = blurred_block
            else:
                filtered[i:min(i+block_size, h), j:min(j+block_size, w)] = block
    
    down_adaptive = cv2.resize(filtered.astype(np.uint8), (w // M_uniform, h // M_uniform), cv2.INTER_NEAREST)
    up_adaptive = cv2.resize(down_adaptive, (w, h), cv2.INTER_LINEAR)   
    # 计算误差
    mse_uniform = np.mean((chess.astype(np.float32) - up_uniform.astype(np.float32))**2)
    mse_adaptive = np.mean((chess.astype(np.float32) - up_adaptive.astype(np.float32))**2)   
    # 误差图
    err_uniform = np.abs(chess.astype(np.float32) - up_uniform.astype(np.float32))
    err_adaptive = np.abs(chess.astype(np.float32) - up_adaptive.astype(np.float32))
    
    plt.figure(figsize=(18, 10))
    
    plt.subplot(2, 4, 1)
    plt.imshow(chess, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(up_uniform, cmap='gray')
    plt.title(f'Uniform sigma={sigma_uniform}\nMSE={mse_uniform:.1f}')
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    plt.imshow(up_adaptive, cmap='gray')
    plt.title(f'Adaptive sigma\nMSE={mse_adaptive:.1f}')
    plt.axis('off')
    
    plt.subplot(2, 4, 4)
    plt.imshow(M_map, cmap='hot')
    plt.title('Local M Map (Gradient-based)')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 4, 5)
    plt.imshow(err_uniform, cmap='hot')
    plt.title(f'Uniform Method Error Map\n(Mean Error={mse_uniform:.1f})')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 4, 6)
    plt.imshow(err_adaptive, cmap='hot')
    plt.title(f'Adaptive Method Error Map\n(Mean Error={mse_adaptive:.1f})')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 4, 7)
    plt.imshow(sigma_map, cmap='hot')
    plt.title('Local sigma Map (0.45 * M)')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 4, 8)
    diff = cv2.absdiff(up_uniform, up_adaptive)
    plt.imshow(diff, cmap='hot')
    plt.title('Difference between Two Methods')
    plt.colorbar()

    plt.axis('off')   
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== Adaptive Downsampling Results ===")
    print(f"Uniform sigma MSE = {mse_uniform:.2f}")
    print(f"Adaptive sigma MSE = {mse_adaptive:.2f}")
    print(f"Improvement = {(mse_uniform - mse_adaptive)/mse_uniform*100:.1f}%")
    print(f"\nNote: Adaptive method uses smaller sigma in high-gradient regions (edges)")
    print(f"      and larger sigma in low-gradient regions (smooth areas)")
    
if __name__ == '__main__':
    print("\n=== Part 1: Aliasing Analysis (Chessboard & Chirp) ===")
    part1()   
    print("\nGenerating chessboard for Part 2 & 3...")
    chessboard = generate_chessboard(size=512, block_size=32)   
    print("\n=== Part 2: Sigma Formula Validation ===")
    part2(chessboard)   
    print("\n=== Part 3: Adaptive Downsampling ===")
    part3(chessboard)