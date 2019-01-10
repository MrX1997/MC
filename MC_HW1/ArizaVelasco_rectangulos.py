# Geometria

a1=1.0 # siempre mayor a cero
c1=1.0 # siempre mayor a cero
x0_1=0.0
y0_1=0.0
xmin_1=x0_1 - a1/2.0 
xmax_1=x0_1 + a1/2.0 
ymin_1=y0_1 - c1/2.0 
ymax_1=y0_1 + c1/2.0 

a2=1.0 # siempre mayor a cero
c2=1.0 # siempre mayor a cero
x0_2=1.0                                                                                                                                    
y0_2=1.0
xmin_2=x0_2 - a2/2.0 
xmax_2=x0_2 + a2/2.0 
ymin_2=y0_2 - c2/2.0 
ymax_2=y0_2 + c2/2.0 



class Rectangle:
    
    def __init__(self, min_x=0, max_x=0, min_y=0, max_y=0):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y


    def intersectacion(self, other):
        if(self.min_x > other.max_x or self.max_x < self.min_x): 
            return('Los rectangulos no se intersectan')

        if(self.min_y > other.max_y or self.max_y < self.min_y): 
            return('Los rectangulos no se intersectan')
            
        if(self.min_x > other.max_x or self.max_x < self.min_x): 
            return('Los rectangulos no se intersectan')
            
        if(self.min_x==other.min_x and self.min_y==other.min_y and self.max_x==other.max_x and self.min_y==other.min_y and self.min_x==other.min_x and self.max_y==other.max_y and self.max_x==other.max_x and self.max_y==other.max_y):
            return('Los rectangulos se intersectan formando un rectangulo de lados', (self.max_x-self.min_x) , (self.max_y-self.min_y)) 
            
        if(self.min_x==other.max_x and self.max_y==other.min_y):
            return('Los rectangulos se tocan en un solo punto de coordenadas', self.min_x , self.max_y) 
    
        if(self.min_x==other.max_x and self.min_y==other.max_y):
            return('Los rectangulos se tocan en un solo punto de coordenadas', self.min_x , self.min_y)
        
        if(self.max_x==other.min_x and self.min_y==other.max_y):
            return('Los rectangulos se tocan en un solo punto de coordenadas', self.max_x , self.min_y)
    
        if(self.max_x==other.min_x and self.max_y==other.min_y):
            return('Los rectangulos se tocan en un solo punto de coordenadas', self.max_x , self.max_y)
    
        if(self.max_y==other.min_y):
            return('Los rectangulos tocan en un segmento de recta horizontal y=', self.max_y)
    
        if(self.min_y==other.max_y):
            return('Los rectangulos tocan en un segmento de recta horizontal y=', self.min_y)
        
        if(self.max_x==other.min_x):
            return('Los rectangulos tocan en un segmento de recta vertical x=', self.max_x)
    
    
        if(self.min_x==other.max_x):
            return('Los rectangulos tocan en un segmento de recta vertical x=', self.min_x)    
        
        min_x = max(self.min_x, other.min_x) 
        max_x = min(self.max_x, other.max_x) 
        min_y = max(self.min_y, other.min_y) 
        max_y = min(self.max_y, other.max_y)
        
        
        return('Los rectangulos se intersectan formando un rectangulo de lados', abs(max_x-min_x) , abs(max_y-min_y)) 
   

r1 = Rectangle(xmin_1,xmax_1,ymin_1,ymax_1)
r2 = Rectangle(xmin_2,xmax_2,ymin_2,ymax_2)
print (r1.intersectacion(r2))
