# 根据缺陷面积判断产品状态
def judge_status(defect_area):
    if defect_area >= 100:
        return "NG"
    else:
        return "OK"


# 原始检测数据：这里只有图片名和缺陷面积，还没有 status
raw_data = [
    {"image_name": "image_01.jpg", "defect_area": 0},
    {"image_name": "image_02.jpg", "defect_area": 120},
    {"image_name": "image_03.jpg", "defect_area": 280},
    {"image_name": "image_04.jpg", "defect_area": 30},
]


# results 用来保存自动生成后的完整检测结果
results = []

# 遍历每一条原始数据
for item in raw_data:
    image_name = item["image_name"]
    defect_area = item["defect_area"]

    # 用函数根据缺陷面积自动判断 OK 或 NG
    status = judge_status(defect_area)

    # 把自动判断后的结果保存成一个新的字典
    result = {
        "image_name": image_name,
        "status": status,
        "defect_area": defect_area
    }

    # 把这一张图片的检测结果添加到 results 列表中
    results.append(result)


# 初始化统计变量
ok_count = 0
ng_count = 0
total_area = 0

# 遍历自动生成后的检测结果，进行统计
for result in results:
    total_area = total_area + result["defect_area"]

    if result["status"] == "OK":
        ok_count = ok_count + 1
    else:
        ng_count = ng_count + 1


# 计算总数量
total_count = len(results)

# 计算 OK 率和 NG 率
ok_rate = ok_count / total_count
ng_rate = ng_count / total_count


# 输出完整检测结果
print(results)

# 输出统计结果
print(f"OK数量：{ok_count}")
print(f"NG数量：{ng_count}")
print(f"总数量：{total_count}")
print(f"总缺陷面积：{total_area}")
print(f"OK率：{ok_rate * 100:.1f}%")
print(f"NG率：{ng_rate * 100:.1f}%")