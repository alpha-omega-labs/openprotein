import datetime

def original_labels_to_fasta(label_list):
    sequence = ""
    for label in label_list:
        if label == 0:
            sequence = sequence + "M"
        if label == 1:
            sequence = sequence + "M"
        if label == 2:
            sequence = sequence + "S"
        if label == 3:
            sequence = sequence + "I"
        if label == 4:
            sequence = sequence + "O"
        if label == 5:
            sequence = sequence + "-"
    return sequence

def post_process_prediction_data(prediction_data):
    data = []
    for (name, aa_string, actual, prediction) in zip(*prediction_data):
        data.append("\n".join([">" + name,
                               aa_string,
                               actual,
                               original_labels_to_fasta(prediction)]))
    return "\n".join(data)


def is_topologies_equal(topology_a, topology_b, minimum_seqment_overlap=5):
    if len(topology_a) != len(topology_b):
        return False
    for idx, (_position_a, label_a) in enumerate(topology_a):
        if label_a != topology_b[idx][1]:
            return False
        if label_a in (0, 1):
            overlap_segment_start = max(topology_a[idx][0], topology_b[idx][0])
            overlap_segment_end = min(topology_a[idx + 1][0], topology_b[idx + 1][0])
            if overlap_segment_end - overlap_segment_start < minimum_seqment_overlap:
                return False
    return True

def get_experiment_id():
    return globals().get("experiment_id")


def write_out(*args, end='\n'):
    output_string = datetime.now().strftime('%Y-%m-%d %H:%M:%S') \
                    + ": " + str.join(" ", [str(a) for a in args]) + end
    if globals().get("experiment_id") is not None:
        with open("output/" + globals().get("experiment_id") + ".txt", "a+") as output_file:
            output_file.write(output_string)
            output_file.flush()
    print(output_string, end="")