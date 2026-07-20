from modelscope.hub.snapshot_download import snapshot_download

# 指定模型 ID
model_id = 'facebook/dinov3-vith16plus-pretrain-lvd1689m'

# 强制指定下载路径为您刚刚挂载的数据盘
local_dir_root = '/data/models'

print("开始下载模型到:", local_dir_root)
# 开始下载，它会返回模型最终所在的本地绝对路径
model_dir = snapshot_download(model_id, cache_dir=local_dir_root)
print("下载完成！模型本地路径为:", model_dir)
