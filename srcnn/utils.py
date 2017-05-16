import tensorflow as tf


def replicate_padding(tensor, pad_amt):
    """
    copy padding
    :param tensor:
    :param pad_amt:
    :return:
    """
    if isinstance(pad_amt, int):
        pad_amt = [pad_amt] * 2
    for axis, p in enumerate(pad_amt):
        tensor = _prepend_edge(tensor, p, axis=axis+1)
        tensor = _append_edge(tensor, p, axis=axis+1)
    return tensor


def _prepend_edge(tensor, pad_amt, axis=1):
    """
    This function is intented to add 'reflective' padding to a 4d Tensor across
    the height and width dimensions
    :param tensor: Tensor with rank 4
    :param pad_amt: Integer
    :param axis: Must be in (1,2)
    :return:
    """
    if axis not in (1, 2):
        raise ValueError("Axis must equal 0 or 1.Axis is set to %i" % axis)

    if axis == 1:
        concat_dim = 2
    else:
        concat_dim = 1

    begin = [0, 0, 0, 0]
    end = [-1, -1, -1, -1]
    end[axis] = 1

    edges = pad_amt*[tf.slice(tensor, begin, end)]
    if len(edges) > 1:
        padding = tf.concat(axis=axis, values=edges)
    else:
        padding = edges[0]

    tensor_padded = tf.concat(axis=axis, values=[padding, tensor])
    return tensor_padded



