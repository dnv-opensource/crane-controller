Assurance case notes
====================

Concept
-------
* Control the crane within
   - the stated environmental limits
   - the stated control limits
* 'crane' refers to simulator or real crane
   - simulator should be assured with respect to
     realistically simulate both crane and environment,
     so that

      + controller tests do not fool the user
      + controller learning is not disturbed by crane simulator errors

   - controller should be assured to perform the control actions with the stated accuracy

      + control goal (reward) specified without 'backdoor'
      + tests performed for all stated environmental conditions
      + warning when actions are attempted outside the stated limits.

Crane limitations
   See crane documentation. E.g. stiff booms

Environment simulator
   The environmental capabilities shall be specified.

   #. crane mounted on fixed structure. External forces directly specified
   #. crane mounted on moveable structure ...

Control actions
   The following control actions are anticipated

   #. anti-pendulum. Bring the load to rest as quick as possible,
      when starting from and load velocity (max ?? m/s)
   #. place load at a given position (closeness measure and velocity when hitting the ground)