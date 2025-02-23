import json
import os
import argparse
from moviepy.editor import VideoFileClip, concatenate_videoclips


def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract subclips from two input videos based on JSON files and merge them.")
    parser.add_argument("video1_dir", type=str, help="Directory containing the first video.")

    return parser.parse_args()


def game_time_to_seconds(game_time):
    # half, time = game_time.split(" - ")
    time = game_time
    minutes, seconds = map(int, time.split(":"))
    total_seconds = minutes * 60 + seconds
    return total_seconds

def merge_intervals(intervals, appended_labels):
    merged = []
    merged_lbls = []
    paired = list(zip(intervals, appended_labels))
    paired_sorted = sorted(paired, key=lambda x: x[0])
    sorted_intervals, sorted_lbls = zip(*paired_sorted)
    sorted_intervals = list(sorted_intervals)
    sorted_lbls = list(sorted_lbls)

    for index, interval in enumerate(sorted_intervals):
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
            merged_lbls.append(sorted_lbls[index])
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]), max(merged[-1][-1], interval[-1]))
            merged_lbls[-1] = merged_lbls[-1] + " " + sorted_lbls[index]
    return merged, merged_lbls


g1_default_thresh = 0.6
g2_thresh = 0.8
default_interval_len = 5

var_thresh = 0.3
var_interval_len = 8
var_action = "Goal"

var2_thresh = 0.75
var2_interval_len = 6
var2_action = "Corner"

var3_thresh = 0.8
var3_interval_len = 5
var3_action = "Clearance"

var4_thresh = 0.6
var4_interval_len = 6
var4_action = "Shots on target"

var5_thresh = 0.6
var5_interval_len = 5
var5_action = "Shots off target"

lbls_intrvls_fh = {}
lbls_intrvls_sh = {}

group1_labels = {
    "Goal", "Penalty", "Corner", "Shots on target", "Direct free-kick", "Indirect free-kick",
    "Yellow card", "Clearance", "Red card", "Yellow->red card", "Shots off target", "Foul", "Offside"
}

group2_labels = {
    "Ball out of play", "Throw-in", "Substitution", "Kick-off"
}

label_thresholds = {
    var_action: var_thresh,
    var2_action: var2_thresh,
    var3_action: var3_thresh,
    var4_action: var4_thresh,
    var5_action: var5_thresh
}

label_intervals = {
    var_action: var_interval_len,
    var2_action: var2_interval_len,
    var3_action: var3_interval_len,
    var4_action: var4_interval_len,
    var5_action: var5_interval_len
}

def visual_merging(first_dir):
    video1_name = os.path.basename(first_dir)

    input_video1_path = os.path.join(first_dir, f"{video1_name}.mp4")

    json1_path = os.path.join(first_dir, "results_spotting.json")

    video1 = VideoFileClip(input_video1_path)
    video1_duration = video1.duration
    max_duration = int(video1_duration * 0.156)

    with open(json1_path, 'r') as f:
        data = json.load(f)

    intervals = []
    lbls = []
    both_vid_duration = 0
    total_duration_vid1 = 0

    for prediction in data["predictions"]:
        boop = 0
        label = prediction.get("label")
        confidence = float(prediction.get("confidence", 0))
        threshold = label_thresholds.get(label, g1_default_thresh)
        interval_length = label_intervals.get(label, default_interval_len)

        if label in group1_labels and confidence > threshold:
            time_in_seconds = game_time_to_seconds(prediction["gameTime"])
            start_interval = max(0, time_in_seconds - interval_length)
            end_interval = min(time_in_seconds + interval_length, video1_duration)
            duration = end_interval - start_interval
            intervals.append((start_interval, end_interval, confidence))
            lbls.append(label)
            total_duration_vid1 += duration

    merged_intervals_vid1, merged_lbls_vid1 = merge_intervals(intervals, lbls)

    g2_predictions_vid1 = [
        (prediction, game_time_to_seconds(prediction["gameTime"]))
        for prediction in data["predictions"]
        if prediction.get("label") in group2_labels and float(prediction.get("confidence", 0)) > g2_thresh
    ]
    g2_predictions_vid1.sort(key=lambda x: float(x[0]["confidence"]), reverse=True)

    both_vid_duration = total_duration_vid1
    g2_vid1_final = []
    vid1_final_lbls = []

    default_g2_interval_len = 5

    if both_vid_duration < max_duration:
        itr1 = 0
        itr2 = 0
        while both_vid_duration < max_duration:
            try:
                start_interval = max(0, g2_predictions_vid1[itr1][1] - default_g2_interval_len)
                end_interval = min(g2_predictions_vid1[itr1][1] + default_g2_interval_len, video1_duration)
                duration = end_interval - start_interval
                itr1 += 1
                g2_vid1_final.append((start_interval, end_interval, g2_predictions_vid1[itr1][0].get("confidence")))
                vid1_final_lbls.append(g2_predictions_vid1[itr1][0].get("label"))
                both_vid_duration += duration
            except:
                break

    final_intervals_vid1, final_labels_vid1 = merge_intervals([(0, 2, 1)] + merged_intervals_vid1 + g2_vid1_final, ['kikoff'] + merged_lbls_vid1 + vid1_final_lbls)

    final_intervals_vid1 = sorted(final_intervals_vid1, key=lambda t: (t[1] - t[0], t[2]), reverse=True)

    return final_intervals_vid1, max_duration

if __name__ == "__main__":
    args = parse_arguments()
    print(visual_merging(args.video1_dir))
