extends RigidBody2D

var area_center

func generate_ground(num_vert, height=150.0 ):
	var polygon = PoolVector2Array()
	var platf_width = 100.0
	var dx = (get_viewport().size.x - platf_width)/(num_vert)
	var shift_x = get_viewport().size.x/2.0
	var rand_x = 10.0
	var rand_y = 100.0
	
	#left part
	for i in range(num_vert/2):
		var x = -shift_x + i*dx + 2.0*(randf()-0.5)*rand_x
		var prev_y = -height
		if i>0:
			prev_y = polygon[-1].y
		var y = prev_y - 2.0*(randf()-0.5)*rand_y 
		polygon.append(Vector2(x, y))
	
	#platform	
	polygon.append(Vector2(-50.0, -height))
	polygon.append(Vector2(50.0, -height))
	
	#right part
	for i in range(1, num_vert/2+1):
		var prev_y = polygon[-1].y
		var x = platf_width/2.0 + i*dx + 2.0*(randf()-0.5)*rand_x
		var y = prev_y - 2.0*(randf()-0.5)*rand_y
		
		polygon.append(Vector2(x, y))
	
	#bottom
	polygon.append(Vector2(shift_x, -height))
	polygon.append(Vector2(shift_x, 0.0))
	polygon.append(Vector2(-shift_x, 0.0))
	return polygon

func _ready():
	var height = 150.0
	var ground = generate_ground(10, height)
	$CollisionPolygon2D.set_polygon(ground)
	$Polygon2D.set_polygon(ground)
	$LandingArea.transform.origin = Vector2(0, -height)
	area_center = transform.basis_xform($LandingArea.transform.origin)
	