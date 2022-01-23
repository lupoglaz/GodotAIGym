extends RigidBody2D


# Declare member variables here. Examples:
# var a = 2
# var b = "text"
var area_center

signal landing_body_entered(body)
signal landing_body_exited(body)

func generate_part(array, num_vert, lr=-1):
	var rand_x = 10.0
	var rand_y = 100.0
	var dx = get_viewport().size.x/num_vert
	var init_x = array[-1].x
	for i in range(1, num_vert+1):
		var x = init_x + i*dx*(lr) + 2.0*(randf()-0.5)*rand_x
		var y = array[-1].y + 2.0*(randf()-0.5)*rand_y
		if y>=-5.0:
			y = -5.0
		if abs(x - array[-1].x) < 5:
			x = array[-1].x + lr*5
		array.append(Vector2(x,y))
	return array

func generate_ground(num_vert, height):
	var polygon = PoolVector2Array()
	var platf_hwidth = $LandingArea/CollisionShape2D.shape.extents.x
	
	polygon.append(Vector2(platf_hwidth, -height))
	polygon.append(Vector2(-platf_hwidth, -height))
	polygon = generate_part(polygon, num_vert/2, -1)
	polygon.invert()
	polygon = generate_part(polygon, num_vert/2, 1)
	polygon.append(Vector2(polygon[-1].x, -height))
	polygon.append(Vector2(polygon[-1].x, 0))
	polygon.append(Vector2(polygon[0].x, 0))
	
	return polygon
	

# Called when the node enters the scene tree for the first time.
func _ready():
	var ground = generate_ground(20, 150.0)
	$CollisionPolygon2D.set_polygon(ground)
	
	$Polygon2D.set_polygon(ground)
	$Polygon2D.set_color(Color(0,0,0))
	
	$LandingArea.transform.origin = Vector2(0, -150.0)
	area_center = transform.xform($LandingArea.transform.origin)
	update()

func _draw():
	var Outline = Color(1,1,1)
	var Width = 2.0
	var poly = $Polygon2D.get_polygon()
	for i in range(1, poly.size()):
		draw_line(poly[i-1], poly[i], Outline, Width)

# Called every frame. 'delta' is the elapsed time since the previous frame.
#func _process(delta):
#	pass


func _on_LandingArea_body_entered(body):
	emit_signal("landing_body_entered", body)


func _on_LandingArea_body_exited(body):
	emit_signal("landing_body_exited", body)
