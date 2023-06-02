class DQN(tf.keras.Model):
    def __init__(self, states, actions):
        super(DQN, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(states,))
        self.hidden_layer = tf.keras.layers.Dense(128)
        self.output_layer = tf.keras.layers.Dense(actions)

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        z = self.hidden_layer(z)
        output = self.output_layer(z)
        return output
