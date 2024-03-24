import matplotlib.pyplot as plt

# 创建画布和子图
fig, axs = plt.subplots(2, 2, figsize=(10, 5))

# 绘制模型结构图
axs[0, 0].text(0.5, 0.5, '模型结构', fontsize=12, ha='center')
axs[0, 0].axis('off')

# 绘制右上角饼图
labels_upper = ['A', 'B', 'C']
sizes_upper = [30, 40, 30]
axs[0, 1].pie(sizes_upper, labels=labels_upper, autopct='%1.1f%%', startangle=90)
axs[0, 1].set_title('上部饼图')

# 绘制右下角饼图
labels_lower = ['X', 'Y', 'Z']
sizes_lower = [20, 30, 50]
axs[1, 1].pie(sizes_lower, labels=labels_lower, autopct='%1.1f%%', startangle=90)
axs[1, 1].set_title('下部饼图')

# 调整子图间距
plt.tight_layout()

plt.show()
plt.savefig('model_structure.png')