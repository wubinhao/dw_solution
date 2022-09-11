
import collections
# import copy
import json
import logging


logger = logging.getLogger(__name__)

class MODE:
    """

    Note:
        Currently StrEnum cannot be imported with the environment
        `sys.version_info < (3, 11)`, so we simply create a MODE class here.
    """
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'
    FINETUNE = 'finetune'


def use_diff(func):
    def wrapper(self, *args, **kwargs):
        if self.cfg.federate.use_diff:
            # TODO: any issue for subclasses?
            before_metric = self.evaluate(target_data_split_name='val')

        num_samples_train, model_para, result_metric = func(
            self, *args, **kwargs)

        if self.cfg.federate.use_diff:
            # TODO: any issue for subclasses?
            after_metric = self.evaluate(target_data_split_name='val')
            result_metric['val_total'] = before_metric['val_total']
            result_metric['val_avg_loss_before'] = before_metric[
                'val_avg_loss']
            result_metric['val_avg_loss_after'] = after_metric['val_avg_loss']

        return num_samples_train, model_para, result_metric

    return wrapper


def format_log_hooks(hooks_set):
    def format_dict(target_dict):
        print_dict = collections.defaultdict(list)
        for k, v in target_dict.items():
            for element in v:
                print_dict[k].append(element.__name__)
        return print_dict

    if isinstance(hooks_set, list):
        print_obj = [format_dict(_) for _ in hooks_set]
    elif isinstance(hooks_set, dict):
        print_obj = format_dict(hooks_set)
    return json.dumps(print_obj, indent=2).replace('\n', '\n\t')


def filter_by_specified_keywords(param_name, filter_keywords):
    '''
    Arguments:
        param_name (str): parameter name.
    Returns:
        preserve (bool): whether to preserve this parameter.
    '''
    preserve = True
    for kw in filter_keywords:
        if kw in param_name:
            preserve = False
            break
    return preserve


