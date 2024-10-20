class Pod:
    def __init__(self, position, velocity, angle, next_checkpoint_id):
        self.position = position
        self.velocity = velocity
        self.angle = angle
        self.next_checkpoint_id = next_checkpoint_id

        self.boost_used = False
        self.checkpoints_passed = 0
