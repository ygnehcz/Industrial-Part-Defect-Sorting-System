# 根据缺陷面积自动判断 OK 或 NG
def judge_status(defect_area):
    if defect_area >= 100:
        return "NG"
    else:
        return "OK"


# 保存每张检测图片的名称和缺陷面积
results = [
    {"image_name": "image_01.jpg", "defect_area": 0},
    {"image_name": "image_02.jpg", "defect_area": 120},
    {"image_name": "image_03.jpg", "defect_area": 280},
    {"image_name": "image_04.jpg", "defect_area": 30},
]

# ok_count：合格数量；ng_count：不合格数量；total_area：所有缺陷面积之和
ok_count = 0
ng_count = 0
total_area = 0

# 遍历每张图片的检测结果，逐个统计
for result in results:
    # 从当前这条记录中取出缺陷面积
    defect_area = result["defect_area"]
    # 调用函数，根据面积自动得到 OK 或 NG
    status = judge_status(defect_area)

    total_area += defect_area
    if status == "OK":
        ok_count += 1
    else:
        ng_count += 1

# 输出统计结果
print(f"OK数量：{ok_count} ")
print(f"NG数量：{ng_count} ")
print(f"总缺陷面积：{total_area} ")
print(f"OK率：{ok_count / len(results) * 100:.1f}%")
print(f"NG率：{ng_count / len(results) * 100:.1f}%")
