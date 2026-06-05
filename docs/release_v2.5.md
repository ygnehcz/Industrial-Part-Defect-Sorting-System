# V2.5 版本说明：最终作品集包装版

## 1. 版本定位

V2.5 是项目的最终作品集包装版本。不新增模型训练，只做项目展示、文档整合和简历材料整理。

## 2. 新增文档

| 文档 | 说明 |
|------|------|
| `docs/final_project_report.md` | 最终项目报告（完整演进路线） |
| `docs/interview_questions.md` | 20 个面试问答 |
| `docs/resume_snippets.md` | 简历一行/三行/STAR 版 + 技术栈关键词 |
| `docs/release_v2.5.md` | 本说明 |

## 3. 当前最佳结果

| 模型 | conf | F1 | Precision | Recall |
|------|------|-----|-----------|--------|
| YOLOv8n-seg 3ep | 0.05 | 0.833 | 90.4% | 77.3% |

## 4. 项目版本路线图

```
V0.x: 模拟图片 + 基础 OpenCV
V1.x: 真实数据 + CLAHE 增强 + 实验可视化
V2.x: YOLO segmentation 训练 + IoU/Dice + 模型选择
V2.5: 作品集包装
```

## 5. 项目完成状态

✅ 2 种方法（OpenCV CLAHE + YOLO segmentation）
✅ 14 个版本标签
✅ 完整文档体系
✅ 100+ 组参数实验
✅ GitHub 展示
✅ 简历与面试材料
