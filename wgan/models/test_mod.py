import tensorflow as tf


def main():

    v_num = -1.1
    v_div = 2.0

    mod_tf = tf.mod((v_num+1.0), v_div) - 1.0 
    mod_py = v_num % v_div

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        print('TF: {} % {} = {}'.format(v_num, v_div, mod_tf.eval()))
        print('PY: {} % {} = {}'.format(v_num, v_div, mod_py))

if __name__ == "__main__":
    main()
