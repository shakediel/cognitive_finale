
(define (problem simple_maze)
(:domain maze)
(:objects
	person1
	start_tile
	c0
	c1
	c2
	c3
	g0
	g1
	g2
	g3
	goal_tile
	d0
	d1
	d2
	d3
	d4
	)
(:init
	(empty start_tile)
	(empty c0)
	(empty c1)
	(empty c2)
	(empty c3)
	(empty g0)
	(empty g1)
	(empty g2)
	(empty g3)
	(empty goal_tile)
	(empty d0)
	(empty d1)
	(empty d2)
	(empty d3)
	(empty d4)
	(east start_tile c0)
	(west c0 start_tile)
	(east c0 c1)
	(west c1 c0)
	(east c1 c2)
	(west c2 c1)
	(east c2 c3)
	(west c3 c2)
	(north g0 g1)
	(south g1 g0)
	(north g1 g2)
	(south g2 g1)
	(north g2 g3)
	(south g3 g2)
	(north g3 goal_tile)
	(south goal_tile g3)
	(south d0 d1)
	(north d1 d0)
	(south d1 d2)
	(north d2 d1)
	(south d2 d3)
	(north d3 d2)
	(south d3 d4)
	(north d4 d3)
	(north c3 g0)
	(south g0 c3)
	(south c3 d0)
	(north d0 c3)
    (person person1)
    (at person1 start_tile)   
        )
(:goals
	(all (at person1 goal_tile)  (at person1 d4))    
	)
)
