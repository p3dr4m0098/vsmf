import json
import argparse
from moviepy.editor import VideoFileClip, concatenate_videoclips


def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract subclips from two input videos based on JSON files and merge them.")
    parser.add_argument("json_path", type=str, help="Directory containing the first video.")
    parser.add_argument("vid_path", type=str, help="Directory containing the first video.")

    return parser.parse_args()

def merge_intervals_nlp(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = []
    for current in intervals:
        if not merged or merged[-1][1] < current[0]:
            merged.append(list(current))
        else:
            merged[-1][1] = max(merged[-1][1], current[1])
            merged[-1][2] = max(merged[-1][2], current[2])
    return merged

def find_scores_in_time_range(data, target_id, th_s=5, th_e=5):
    target_segment = None
    for item in data:
        segments = item.get("segments", [])
        for segment in segments:
            if segment.get("id") == target_id:
                target_segment = segment
                break
        if target_segment:
            break

    if not target_segment:
        return []

    start_time = convert_to_seconds(target_segment['start'])
    end_time = convert_to_seconds(target_segment['end'])

    mid = (start_time + end_time) / 2

    time_range_start = mid - th_s #start_time - th_s
    time_range_end = mid + th_e #end_time + th_e

    scores = []
    for item in data:
        segments = item.get("segments", [])
        score_f = 0
        score_s = 0
        for segment in segments:
            segment_start = convert_to_seconds(segment['start'])
            segment_end = convert_to_seconds(segment['end'])
            if (segment_start <= time_range_end and segment_end >= time_range_start) or (segment['id'] == target_id):
                score_f += segment['score_f']
                score_s += segment['score_s']

        scores.append(score_f)
        scores.append(score_s)

    return scores

def convert_to_seconds(time_str):
    hours, minutes, seconds = map(float, time_str.split(':'))
    return hours * 3600 + minutes * 60 + seconds

def add_features(data,first_priority,second_priority):
    for item in data:
        segments = item.get("segments", [])
        for index, segment in enumerate(segments):
            segment["id"] = index

            wf_list = [w for w in first_priority if w in segment["text"]]
            ws_list = [w for w in second_priority if w in segment["text"]]

            counter_f = len(wf_list)
            counter_s = len(ws_list)

            segment["wf_list"] = wf_list
            segment["ws_list"] = ws_list

            segment["score_f"] = counter_f
            segment["score_s"] = counter_s

    return data

def process_json(data, th_s=5, th_e=5, duration=0):
    result = {}
    result_first = []
    result_second = []
    for item in data:
        segments = item['segments']
        for segment in segments:
            if segment["wf_list"]:

                start_seconds = max(convert_to_seconds(segment["start"]) - th_s, 0)
                end_seconds = min(convert_to_seconds(segment["end"]) + th_e, duration)
                scores = find_scores_in_time_range(data,segment['id'], th_s, th_e)

                if any(word in segment["wf_list"] for word in [" گل ", "فرصت", "باز شد","توی دروازه"]):
                    result_first.append(["گل", scores[0], start_seconds, end_seconds])

                elif "موقعیت" in segment["wf_list"]:
                    result_first.append(["موقعیت", scores[0], start_seconds, end_seconds])

                elif "خطرناک" in segment["wf_list"]:
                    result_first.append(["خطرناک", scores[0], start_seconds, end_seconds])

                elif "کارت قرمز" in segment["wf_list"]:
                    result_first.append(["کارت قرمز", scores[0], start_seconds, end_seconds])

                elif "پنالتی" in segment["wf_list"]:
                    result_first.append(["پنالتی", scores[0], start_seconds, end_seconds])

                else:
                    continue

            if segment["ws_list"]:
                start_seconds = max(convert_to_seconds(segment["start"]) - th_s, 0)
                end_seconds = min(convert_to_seconds(segment["end"]) + th_e, duration)
                scores = find_scores_in_time_range(data,segment['id'], th_s, th_e)

                if any(word in segment["ws_list"] for word in ["آفساید","افساید"]):
                    result_second.append(["آفساید", scores[1], start_seconds, end_seconds])

                elif "کورنر" in segment["ws_list"]:
                    result_second.append(["کورنر", scores[1], start_seconds, end_seconds])

                elif "ضد حمله" in segment["ws_list"]:
                    result_second.append(["ضد حمله", scores[1], start_seconds, end_seconds])

                elif "خطا" in segment["ws_list"]:
                    result_second.append(["خطا", scores[1], start_seconds, end_seconds])

                elif "کارت زرد" in segment["ws_list"]:
                    result_second.append(["کارت زرد", scores[1], start_seconds, end_seconds])

                elif "تعویض" in segment["ws_list"]:
                    result_second.append(["تعویض", scores[1], start_seconds, end_seconds])
                else:
                    continue

    result["first_priority"] = result_first
    result["second_priority"] = result_second

    return result


def process_video(results):

    intervals = []

    for item in results["first_priority"]:
        start = item[2]
        end = item[3]
        intervals.append((start, end, item[1]))

    new_intervals = merge_intervals_nlp(intervals)

    sp_intervals = []

    for item in results["second_priority"]:
        start = item[2]
        end = item[3]
        sp_intervals.append((start, end, item[1]))

    new_sp_intervals = merge_intervals_nlp(sp_intervals)

    for start, end, scor in new_sp_intervals:
        tmp_st = start
        tmp_en = end
        score = scor
        for st, en, sco in new_intervals:
            if st <= end and en >= start:
                tmp_st = min(start, st)
                tmp_en = max(end, en)
                score = max(scor, sco)
        new_intervals.append((tmp_st, tmp_en, score))
        new_intervals = merge_intervals_nlp(intervals)

    return new_intervals


def NLP_merging(data, vid_path):  
    video = VideoFileClip(vid_path)

    th_s = 10
    th_e = 5

    first_priority = [" گل ", "کارت قرمز", " پنالتی ", "فرصت", "باز شد", "موقعیت","توی دروازه", "خطرناک"]
    second_priority = ["کورنر","آفساید","افساید","ضد حمله","خطا","تعویض",  "کارت زرد"]


    new_data = add_features(data, first_priority, second_priority)
    results = process_json(new_data, th_s = th_s, th_e = th_e, duration = video.duration)

    final_int = process_video(results)

    final_int = sorted(final_int, key=lambda t: (t[1] - t[0], t[2]), reverse=True)

    return final_int



if __name__ == "__main__":
    args = parse_arguments()
    json_file = args.json_path

    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    print(NLP_merging(data, args.vid_path))
