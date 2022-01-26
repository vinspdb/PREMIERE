import pandas as pd
import numpy as np
import itertools
import utility as ut
import csv
import argparse


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Inception for next activity prediction.')
        parser.add_argument('-event_log', type=str, help="Event log name")
        args = parser.parse_args()
        namedataset = args.event_log

        df_fold = pd.read_csv('fold/'+namedataset+'.txt', header=None) #,encoding='windows-1252')
        df_fold.columns = ["CaseID", "Activity", "Resource", "Timestamp"]
        cont_trace = df_fold['CaseID'].value_counts(dropna=False)
        max_trace = max(cont_trace)
        df_fold['Resource'] = df_fold['Resource'].fillna('-1')

        unique_events = df_fold['Activity'].nunique()
        unique_resources = df_fold['Resource'].nunique()

        listOfevents = df_fold['Activity'].unique()
        listOfeventsInt = list(range(1, unique_events + 1))
        mapping = dict(zip(listOfevents, listOfeventsInt))

        df_fold.Activity = [mapping[item] for item in df_fold.Activity]

        listOfres = df_fold['Resource'].unique()
        listOfresInt = list(range(1, unique_resources + 1))
        mapping_res = dict(zip(listOfres, listOfresInt))

        df_fold.Resource = [mapping_res[item] for item in df_fold.Resource]

        # group by activity, resource and timestamp by caseid
        act = df_fold.groupby('CaseID', sort=False).agg({'Activity': lambda x: list(x)})
        res = df_fold.groupby('CaseID', sort=False).agg({'Resource': lambda x: list(x)})
        temp = df_fold.groupby('CaseID', sort=False).agg({'Timestamp': lambda x: list(x)})

        time_prefix = ut.get_time(temp, max_trace)
        i = 0
        time_prefix_new = []
        while i < len(time_prefix):
            time_val = [x for x in time_prefix[i] if x != 0.0]
            time_prefix_new.append(time_val)
            i = i + 1

        sequence_prefix = ut.get_sequence(act,max_trace)
        resource_prefix = ut.get_sequence(res, max_trace)

        i = 0
        list_sequence_prefix = []
        list_resource_prefix = []

        while i < len(sequence_prefix):
            list_sequence_prefix.append(list(np.trim_zeros(sequence_prefix[i])))
            list_resource_prefix.append(list(np.trim_zeros(resource_prefix[i])))
            i = i + 1

        i = 0
        agg_time_feature = []
        while i < len(time_prefix_new):
            time_feature = []
            duration = time_prefix_new[i][-1] - time_prefix_new[i][0]
            time_feature.append(duration.days/86400)
            time_feature.append(len(list_sequence_prefix[i]))
            if len(list_sequence_prefix[i]) == 1:
                time_feature.append(0)
                time_feature.append(0)
                time_feature.append(0)
                time_feature.append(0)
            else:
                diff_cons = [y-x for x,y in ut.pairwise(time_prefix_new[i])]
                diff_cons_sec = [(item.days/86400) for item in diff_cons]
                time_feature.append(np.mean(diff_cons_sec))
                time_feature.append(np.median(diff_cons_sec))
                time_feature.append(np.min(diff_cons_sec))
                time_feature.append(np.max(diff_cons_sec))

            agg_time_feature.append(time_feature)

            i = i + 1

        flow_act = [p for p in itertools.product(listOfeventsInt, repeat=2)]
        target = ut.get_label(act, max_trace)
        premiere_feature = ut.premiere_feature(list_sequence_prefix, list_resource_prefix, flow_act, agg_time_feature, unique_events, unique_resources, target)

        with open("fold/"+namedataset+"feature"+".csv", "w", newline='') as f:
            writer = csv.writer(f, delimiter=',')
            for feature in premiere_feature:
                writer.writerow(ut.output_list(feature))

        print("feature generation complete")
