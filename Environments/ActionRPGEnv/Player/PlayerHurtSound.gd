extends AudioStreamPlayer


func _ready():
	self.connect("finished", self, "queue_free()")
