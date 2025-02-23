def find_overlapping_intervals(list1, list2, vid_duration):

    result = []
    max_du_ = vid_duration
    vid_du_ = 0
    for st1, en1, s1 in list1:
        if vid_du_ > max_du_:
            break
        for st2, en2, s2 in list2:
            if st2 <= en1 and en2 >= st1:
                overlap_start = min(st1, st2)
                overlap_end = max(en1, en2)
                if result:
                    temp = []
                    for res_st, res_en in result:
                        if res_st <= overlap_end and res_en >= overlap_start:
                            overlap_end = max(res_en, overlap_end)
                            overlap_start = min(res_st, overlap_start)
                        else:
                            temp.append((res_st, res_en))
                    temp.append((overlap_start, overlap_end))
                    result = temp.copy()
                else:
                    result.append((overlap_start, overlap_end))
        if result:
            vid_du_ = sum([en_res - st_res for st_res, en_res in result])

    if vid_du_ <= max_du_:
        not_overlapped = []
        for st1, en1, s1 in list1:
            if not any(st1 < en2 and en1 > st2 for st2, en2, _ in list2):
                not_overlapped.append((st1, en1, s1))

        for st2, en2, s2 in list2:
            if not any(st2 < en1 and en2 > st1 for st1, en1, _ in list1):
                not_overlapped.append((st2, en2, s2))

        not_overlapped = sorted(not_overlapped, key=lambda t: (t[1] - t[0], t[2]), reverse=True)
        for st, en, _ in not_overlapped:
            if vid_du_ <= max_du_:
                result.append((st, en))
                vid_du_ += en - st
            else:
                break

    result.sort(key=lambda x: x[0])

    return result


if __name__ == "__main__":
    list1 = []
    list2 = []
    video_duration = 0
    print(find_overlapping_intervals(list1, list2, video_duration))
