extends RigidBody2D

var area_center

var LANDING_HEIGHT = 150.0

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
		var prev_x = -shift_x
		if i>0:
			prev_y = polygon[-1].y
			prev_x = polygon[-1].x
		var y = prev_y - 2.0*(randf()-0.5)*rand_y 
		if y >= -5.0:
			y = -5.0
		if x <= prev_x + 5:
			x = prev_x + 5
		
		polygon.append(Vector2(x, y))
	
	#platform	
	polygon.append(Vector2(-50.0, -height))
	polygon.append(Vector2(50.0, -height))
	
	#right part
	for i in range(1, num_vert/2+1):
		var prev_y = polygon[-1].y
		var prev_x = polygon[-1].x
		var x = platf_width/2.0 + i*dx + 2.0*(randf()-0.5)*rand_x
		var y = prev_y - 2.0*(randf()-0.5)*rand_y
		if y >= -5.0:
			y = -5.0
		if x <= prev_x + 5:
			x = prev_x + 5
		if x >= shift_x:
			break
		
		polygon.append(Vector2(x, y))
	
	#bottom
	polygon.append(Vector2(shift_x, -height))
	polygon.append(Vector2(shift_x, 0.0))
	polygon.append(Vector2(-shift_x, 0.0))
	return polygon

func create_ground():
	shape_owner_clear_shapes(0)
	var ground = generate_ground(10, LANDING_HEIGHT)
	var shape = ConcavePolygonShape2D.new()
	var segments = []
	for i in range(len(ground)-1):
		segments.append(ground[i])
		segments.append(ground[i+1])
	shape.set_segments(segments)
	shape_owner_add_shape(0, shape)
	$Polygon2D.set_polygon(ground)
	$Polygon2D.set_color(Color(0,0,0))
	$LandingArea.transform.origin = Vector2(0, -LANDING_HEIGHT)
	area_center = transform.xform($LandingArea.transform.origin)
	update()

func _ready():
	create_ground()
	
func _draw():
	var OutLine = Color(1,1,1)
	var Width = 2.0
	var poly = $Polygon2D.get_polygon()
	for i in range(1 , poly.size()):
		draw_line(poly[i-1] , poly[i], OutLine , Width)
	#$Polygon2D.draw_line(poly[poly.size() - 1] , poly[0], OutLine , Width)
