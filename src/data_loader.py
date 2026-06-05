import csv


def load_sample_data(csv_path):
    raw_data = []

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            item = {
                "image_name": row["image_name"],
                "defect_area": int(row["defect_area"]),
                "true_status": row["true_status"]
            }

            raw_data.append(item)

    return raw_data