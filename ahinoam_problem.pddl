
(define (problem ahinoam-problem)
(:domain maze)
(:objects
	person
	start_tile
	a1
	a2
	a3
	a4
	c1
	c2
	c3
	c4
	c5
	c6
	c7
	c8
	c9
	c10
	goal_tile
	)
(:fails ((at person start_tile) (move-east) 1)
        ((at person a1) (move-east) 1)
        ((at person a2) (move-east) 1)
        ((at person a3) (move-east) 1)
        ((at person a4) (move-east) 1)

)
(:init
	(empty start_tile)
	(empty a1)
	(empty a2)
	(empty a3)
	(empty a4)
	(empty c1)
	(empty c2)
	(empty c3)
	(empty c4)
	(empty c5)
	(empty c6)
	(empty c7)
	(empty c8)
	(empty c9)
	(empty c10)
	(empty goal_tile)
	(east start_tile a1)
	(west a1 start_tile)
	(east a1 a2)
	(west a2 a1)
	(east a2 a3)
	(west a3 a2)
	(east a3 a4)
	(west a4 a3)
	(east a4 goal_tile)
	(west goal_tile a4)

	(north start_tile c1)
	(south c1 start_tile)
	(north c1 c2)
    (south c2 c1)
    (north c2 c3)
    (south c3 c2)

    (east c3 c4)
	(west c4 c3)
	(east c4 c5)
	(west c5 c4)
	(east c5 c6)
	(west c6 c5)
	(east c6 c7)
	(west c7 c6)
	(east c7 c8)
	(west c8 c7)

	(north goal_tile c10)
	(south c10 goal_tile)
	(north c10 c9)
    (south c9 c10)
    (north c9 c8)
    (south c8 c9)

    (person person)
    (at person start_tile)
)
(:goal 
    (and (at person goal_tile))
	)
)
