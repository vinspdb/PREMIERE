import itertools
import argparse
import pickle
from PREMIERE import PREMIERE

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Inception for next activity prediction.')
        parser.add_argument('-event_log', type=str, help="Event log name")
        args = parser.parse_args()
        dataset = 'receipt'#args.event_log

        pm = PREMIERE(dataset)
        log = pm.import_log()

        max_trace, n_caseid, n_activity = pm.dataset_summary(log=log)
        dict_card = {}
        n_col = 0
        for col in list(log.columns):
            if col != 'case' and col != 'timestamp':
                dict_card[col] = log[col].nunique()
                n_col = n_col + log[col].nunique()

        listOfevents = log['activity'].unique()
        listOfeventsInt = list(range(1, dict_card['activity'] + 1))
        train, test = pm.generate_prefix_trace(log=log)

        dict_view_train = {}
        dict_view_test = {}

        for col in list(log.columns):
            if col != 'case':
                if col == 'timestamp':
                    dict_view_train[col] = pm.get_time(train.groupby('case', sort=False).agg({col: lambda x: list(x)}))
                    dict_view_test[col] = pm.get_time(test.groupby('case', sort=False).agg({col: lambda x: list(x)}))

                else:
                    dict_view_train[col] = pm.get_sequence(train.groupby('case', sort=False).agg({col: lambda x: list(x)}))
                    dict_view_test[col] = pm.get_sequence(test.groupby('case', sort=False).agg({col: lambda x: list(x)}))

        agg_time_train = pm.agg_time_feature(dict_view_train['timestamp'])
        agg_time_test = pm.agg_time_feature(dict_view_test['timestamp'])

        flow_act = [p for p in itertools.product(listOfeventsInt, repeat=2)]

        target_train = pm.get_label(train.groupby('case', sort=False).agg({'activity': lambda x: list(x)}))
        target_test = pm.get_label(test.groupby('case', sort=False).agg({'activity': lambda x: list(x)}))

        premiere_feature_train = pm.premiere_feature(dict_view_train, flow_act, agg_time_train, target_train, dict_card)
        premiere_feature_test = pm.premiere_feature(dict_view_test, flow_act, agg_time_test, target_test, dict_card)

        pm.generate_feature(dataset,premiere_feature_train, 'train')
        pm.generate_feature(dataset,premiere_feature_test, 'test')

        with open("image/" +dataset+"/"+ dataset + "_train_y.pkl", 'wb') as handle:
            pickle.dump(target_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("image/" +dataset+"/"+ dataset + "_test_y.pkl", 'wb') as handle:
            pickle.dump(target_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("feature generation complete")
