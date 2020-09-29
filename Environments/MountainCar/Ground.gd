extends RigidBody2D

var scale_to 
var scale_out
var mini_flag 
var maxi_car

var car
func curve(x):
	return sin(x)*.15+.55

func create_ground():
	shape_owner_clear_shapes(0)
	var polygon = PoolVector2Array()
	scale_to = 6/get_viewport().size.x
	scale_out = get_viewport().size.y - 50
	mini_flag = Vector2(0, get_viewport().size.y)
	maxi_car = Vector2(0, 0)
	
	for i in range(get_viewport().size.x):
		var point = curve(i*scale_to)*scale_out
		polygon.append(Vector2(i,point))
		if point < mini_flag.y:
			mini_flag = Vector2(i, point) 
		if point > maxi_car.y:
			maxi_car = Vector2(i, point)
	
	var shape = ConcavePolygonShape2D.new()
	var segments = []
	for i in range(len(polygon)-1):
		segments.append(polygon[i])
		segments.append(polygon[i+1])
	shape.set_segments(segments)
	shape_owner_add_shape(0, shape)
	shape_owner_add_shape(0, shape)
	$Polygon2D.set_polygon(polygon)
	$Polygon2D.set_color(Color(0,0,0))
	car.transform.origin = maxi_car - Vector2(0, 30)
	$Goal.transform.origin = mini_flag - Vector2(0, $Goal/Sprite.texture.get_size().y/2 -10)
	update()
	

	

# Called when the node enters the scene tree for the first time.
func _ready():
	car = get_node('/root/Enviroment/Car')
	create_ground()


func _draw():
	var OutLine = Color(1,1,1)
	var Width = 2.0
	var poly = $Polygon2D.get_polygon()
	for i in range(1 , poly.size()):
		draw_line(poly[i-1] , poly[i], OutLine , Width)
	#$Polygon2D.draw_line(poly[poly.size() - 1] , poly[0], OutLine , Width)
