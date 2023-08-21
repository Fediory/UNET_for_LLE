import matplotlib.pyplot as plt

# 假设你已经有三组数据：lamda_data, psnr_data, ssim_data
lamda_data = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.20]
psnr_data = [17.308, 17.860, 18.178, 18.342, 18.444, 18.517, 18.546, 18.484, 18.302, 18.007]
ssim_data = [0.811, 0.825, 0.836, 0.845, 0.852, 0.856, 0.858, 0.857, 0.854, 0.849]
# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10))


    
# 绘制 PSNR 数据
ax1.plot(lamda_data, psnr_data, label='PSNR', marker='o', linestyle='-', color='b')
ax1.set_xlabel('Lamda')
ax1.set_ylabel('PSNR')
ax1.set_title('PSNR vs. Lamda')
ax1.grid(True)
ax1.legend()

# 绘制 SSIM 数据
ax2.plot(lamda_data, ssim_data, label='SSIM', marker='x', linestyle='--', color='r')
ax2.set_xlabel('Lamda')
ax2.set_ylabel('SSIM')
ax2.set_title('SSIM vs. Lamda')
ax2.grid(True)
ax2.legend()

# 在每个坐标点上添加 PSNR 的 y 轴值
for i, j in zip(lamda_data, psnr_data):
    ax1.text(i, j, f'{j:.3f}', ha='left', va='bottom', fontsize=8)
    
# 在每个坐标点上添加 SSIM 的 y 轴值
for i, j in zip(lamda_data, ssim_data):
    ax2.text(i, j, f'{j:.2f}', ha='left', va='bottom', fontsize=8)
    
# 调整子图之间的间距
plt.tight_layout()

# 显示图表
plt.show()