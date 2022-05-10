"""Contains class for calculating the statistics of grains - 2d raster images"""
# Imports
from sre_constants import MIN_REPEAT
from statistics import mean
import numpy as np
import skimage.filters as skimage_filters
import skimage.measure as skimage_measure
import skimage.feature as skimage_feature
import scipy.ndimage
import plottingfuncs
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd


class GrainStats:
    """Class for calculating grain stats"""

    def __init__(self,
                 data: np.ndarray,
                 labelled_data: np.ndarray,
                 pixel_to_nanometre_scaling: float,
                 path: Path,
                 filename: str):
        
        """Initialise the class.

        Parameters 
        ----------
        

        """

        self.data = data
        self.labelled_data = labelled_data
        self.savepath = path
        self.pixel_to_nanometre_scaling = pixel_to_nanometre_scaling
        self.filename = filename

        self.calculate_stats()

    
    def calculate_stats(self):

        # Calculate region properties
        region_properties = skimage_measure.regionprops(self.labelled_data)

        # Plot labelled data
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(self.labelled_data, interpolation='nearest', cmap='afmhot')

        stats_array = []
        for index, region in enumerate(region_properties):

            # Create directory for each grain
            path_grain = self.savepath / ('_' + str(index))
            Path.mkdir(path_grain, parents=True, exist_ok=True)

            # Obtain and plot the cropped grain mask
            grain_mask = np.array(region.image)
            plottingfuncs.plot_and_save(grain_mask, path_grain / 'grainmask.png')

            # Obtain the cropped grain image
            minr, minc, maxr, maxc = region.bbox
            grain_image = self.data[minr:maxr, minc:maxc]
            plottingfuncs.plot_and_save(grain_image, path_grain / 'grain_image_raw.png')
            grain_image = np.ma.masked_array(grain_image, mask=np.invert(grain_mask), fill_value=np.nan).filled()
            plottingfuncs.plot_and_save(grain_image, path_grain / 'grain_image.png')
                        

            # Calculate the stats

            min_height = np.nanmin(grain_image)
            max_height = np.nanmax(grain_image)
            median_height = np.nanmedian(grain_image)
            mean_height = np.nanmean(grain_image)

            #volume = np.nansum(grain_image-min_height) 
            volume = np.nansum(grain_image)

            area = region.area
            area_cartesian_bbox = region.area_bbox

            edges = self.calculate_edges(grain_mask)
            min_radius, max_radius, mean_radius, median_radius, centroid = self.calculate_radius_stats(edges)
            hull, hull_indices, hull_simplexes = self.convex_hull(grain_mask, edges, path_grain)
            smallest_bounding_width, smallest_bounding_length, aspect_ratio = self.calculate_aspect_ratio(edges, hull, hull_indices, hull_simplexes, path_grain)

            center_x = minr + centroid[0]
            center_y = minc + centroid[1]

            save_format = '.4f'

            stats = {
                'filename' :                        self.filename,
                'grain no' :                        index,
                'grain_center_x' :                    center_y* self.pixel_to_nanometre_scaling*1e-9,
                'grain_center_y' :                    center_x* self.pixel_to_nanometre_scaling*1e-9,
                'min_radius' :                  min_radius * self.pixel_to_nanometre_scaling*1e-9,
                'max_radius' :                  max_radius * self.pixel_to_nanometre_scaling*1e-9, 
                'grain_mean_radius' :                 mean_radius * self.pixel_to_nanometre_scaling*1e-9,
                'median_radius' :               median_radius * self.pixel_to_nanometre_scaling*1e-9, 
                'grain_minimum' :                  min_height*1e-9, 
                'grain_maximum' :                  max_height*1e-9, 
                'grain_median' :               median_height*1e-9,
                'grain_mean' :                 mean_height*1e-9, 
                'grain_zero_volume' :                      volume * self.pixel_to_nanometre_scaling**2 * 1e-27, 
                'grain_proj_area':                         area * self.pixel_to_nanometre_scaling**2 *1e-18, 
                'area_cartesian_bbox' :         area_cartesian_bbox * self.pixel_to_nanometre_scaling**2, 
                'grain_min_bound_size' :     smallest_bounding_width * self.pixel_to_nanometre_scaling*1e-9, 
                'grain_max_bound_size' :    smallest_bounding_length * self.pixel_to_nanometre_scaling*1e-9, 
                'smallest_bounding_area' :      smallest_bounding_length * smallest_bounding_width * self.pixel_to_nanometre_scaling**2, 
                'aspect_ratio' :                format(aspect_ratio, save_format)
            }
            stats_array.append(stats)

            # Add cartesian bounding box to the stats
            min_row, min_col, max_row, max_col = region.bbox
            rectangle = mpl.patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                                                fill=False, edgecolor='white', linewidth=2)
            ax.add_patch(rectangle)

        grainstats = pd.DataFrame(data=stats_array)
        grainstats.to_csv(self.savepath / 'grainstats.csv')

        plt.savefig(self.savepath / 'labelled_image_bboxes.png')
        plt.close()

    def calculate_radius_stats(self, edges: list):
        edges = np.array(edges)
        centroid = (sum(edges[:, 0]) / len(edges), sum(edges[:, 1]) / len(edges))
        #centroid = (max(edges[:,0])-min(edges[:,0]))/2 , (max(edges[:,1])-min(edges[:,1]))/2

        #centroid = centre
        displacements = edges - centroid
        radii = [radius for radius in np.sqrt(displacements[:, 0]**2 + displacements[:, 1]**2)]
        mean_radius = np.mean(radii)
        median_radius = np.median(radii)
        max_radius = np.max(radii)
        min_radius = np.min(radii) 

        return min_radius, max_radius, mean_radius, median_radius, centroid

    def calculate_edges(self, grain_mask: np.ndarray):
        # #Get outer edge of the grain
        # filled_grain_mask = scipy.ndimage.binary_fill_holes(grain_mask)
        # contours = skimage_measure.find_contours(filled_grain_mask, 1)

        # plottingfuncs.plot_and_save(filled_grain_mask, path / 'filled_mask.png')

        # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        # for index, contour in enumerate(contours):
        #     # plottingfuncs.plot_and_save(contour, path / str('edges' + str(index) + '.png'))
            
        #     ax.plot(contour[0, :], contour[1, :])
        # plt.savefig( path / 'edges.png')
        # plt.close()

        # Fill any holes
        filled_grain_mask = scipy.ndimage.binary_fill_holes(grain_mask)
        
        # Get outer edge using canny filters
        edges = skimage_feature.canny(filled_grain_mask, sigma=3)

        # Plot the edges
        # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        # ax.imshow(edges)
        # plt.savefig(path / 'edges.png')
        # plt.close()




        # print(f'contours: {contours}')
        # np.savetxt(path / 'contours.txt', contours)

        # # Save the contours
        # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        # ax.scatter(x=contours[:, 0], y=contours[:, 1])
        # plt.savefig(path / 'contours.png')


        # Get vectors for the nonzero points in the image
        # points = grain_mask.nonzero()
        # print(f'point: {points[0, 0]}')
        # print(f'points: {points}')
        
        # Save the grain mask
        # np.savetxt(path / 'grainmask.txt', grain_mask)

        nonzero_coordinates = edges.nonzero()
        # Get vector representation of the points 
        # points = list(np.transpose(nonzero_coordinates))

        edges = []
        for vector in np.transpose(nonzero_coordinates):
            edges.append(list(vector))

        # print(f'points: {points}')

        return edges


    def convex_hull(self, grain_mask: np.ndarray, edges: list, path: Path):
        

        # Imports 
        from random import randint

        # def generate_points(number, min=0, max=50):
        #     return [[randint(min, max), randint(min, max)] for i in range(number)]

        

        def plot(coordinates, convex_hull=None, fname=None):
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            xs, ys = zip(*coordinates)
            ax.scatter(xs, ys)
            if convex_hull != None:
                for i in range(1, len(convex_hull)+1):
                    if i == len(convex_hull): i = 0
                    point1 = convex_hull[i - 1]
                    point2 = convex_hull[i]
                    plt.plot((point1[0], point2[0]), (point1[1], point2[1]), '#994400')
            plt.savefig(fname)
            plt.close()

        def get_angle(point1, point2):
            # Function to calculate the angle between two points
            delta_x = point1[0] - point2[0]
            delta_y = point1[1] - point2[1]
            angle = np.arctan2(delta_y, delta_x)
            return angle

        def get_displacement(point1, point2 = None):
            # Get the distance squared between two points. If the second point is not provided, use the starting point.
            if point2 == None: point2 = start_point
            delta_x = point1[0] - point2[1]
            delta_y = point1[1] - point2[1]
            distance_squared = delta_x**2 + delta_y**2
            # Don't need the sqrt since just sorting for dist
            return distance_squared

        def is_clockwise(p1, p2, p3):
            # Determine if three points form a clockwise or counter-clockwise turn.
            M = np.array((  ( p1[0], p1[1], 1),
                            ( p2[0], p2[1], 1), 
                            ( p3[0], p3[1], 1) ))
            if np.linalg.det(M) > 0:
                return False
            else:
                return True

        def sort_points(points):
            # Algorithm based on quicksort to sort the points in order of angle made with the starting point
            if len(points) <= 1: return points
            smaller, equal, larger = [], [], []
            # Get a random point in the array to calculate the pivot angle from
            pivot_angle = get_angle(points[randint(0, len(points)-1)], start_point)
            for point in points:
                point_angle = get_angle(point, start_point)
                if point_angle < pivot_angle: smaller.append(point)
                elif point_angle == pivot_angle: equal.append(point)
                else: larger.append(point)
            # Return sorted array where equal angle points are sorted by distance
            return sort_points(smaller) + sorted(equal, key=get_displacement) + sort_points(larger)


        def graham_scan(points):
            global start_point

            # Find a point guaranteed to be on the hull, here this is the bottom left most point in the array.
            min_y_index = None
            for index, point in enumerate(points):
                if min_y_index == None or point[1] < points[min_y_index][1]:
                    min_y_index = index
                if point[1] == points[min_y_index][1] and point[0] < points[min_y_index][0]:
                    min_y_index = index
                    
            start_point = points[min_y_index]
            points_sorted_by_angle = sort_points(points)

            # Remove starting point from the list so it's not added more than once to the hull
            start_point_index = points_sorted_by_angle.index(start_point)
            del points_sorted_by_angle[start_point_index]
            # Add start point and the first point sorted by angle. Both of these points will always be on the hull.
            hull = [start_point, points_sorted_by_angle[0]]

            for index, point in enumerate(points_sorted_by_angle[1:]):
                # Determine if the proposed point demands a clockwise rotation
                while is_clockwise(hull[-2], hull[-1], point) == True:
                    # Delete the failed point
                    del hull[-1]
                    if len(hull) < 2: break
                hull.append(point)
                # plot(points, hull)

            # Get hull indices from original points array
            hull_indices = []
            for point in hull:
                hull_indices.append(points.index(point))

            # Create simplexes from the hull points
            simplexes = []
            for i in range(len(hull_indices)): 
                simplexes.append((hull_indices[i-1], hull_indices[i]))


            return hull, hull_indices, simplexes

       

        # print(f'len points: {len(points)}')
        hull, hull_indices, simplexes = graham_scan(edges)

         # points = [[randint(0, 50), randint(0, 50)] for i in range(5)]

        # Plot the hull and points
        #plot(edges, hull, path / '_points_hull.png')

        # print(f'points: {points}')
        # print(f'hull: {hull}')
        # print(f'hull indexes: {hull_indices}')
        # print(f'simplexes: {simplexes}')

        # # Convert back to numpy array since numpy is used extensively in finding the minimum bounding rectangle
        # points = np.array(points)

        return hull, hull_indices, simplexes
        
        

    def calculate_aspect_ratio(self, edges: list, convex_hull: np.ndarray, hull_indices: np.ndarray, hull_simplexes: np.ndarray, path: Path):
        
        # # Plot the hull
        # plt.scatter(x=points[:, 0], y=points[:, 1])
        # # Plot simplexes
        # for simplex in simplexes:
        #     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
        edges = np.array(edges)

        smallest_bounding_area = None
        smallest_bounding_rectangle = None

        for simplex in hull_simplexes:
            p1 = edges[simplex[0]]
            p1_index = simplex[0]
            # print(f'p1 index: {p1_index}')
            p2 = edges[simplex[1]]
            p2_index = simplex[1]
            # print(f'p2 index: {p2_index}')
            # print(f'p1: {p1}, p2: {p2}')
            delta = p1 - p2
            angle = np.arctan2(delta[0], delta[1])
            # print(f'angle: {angle*180 / np.pi}')

            # Find the centroid of the points
            centroid = (sum(edges[:, 0]) / len(edges), sum(edges[:, 1] / len(edges)))
            # print(f'centroid: {centroid}')

            # Map all points to the centroid
            remapped_points = edges - centroid
            # plt.scatter(x=remapped_points[:, 0], y=remapped_points[:, 1])
            # plt.plot(remapped_points[simplex, 0], remapped_points[simplex, 1], 'k-')

            # Rotate the coordinates using a rotation matrix
            R = np.array(( (np.cos(angle), -np.sin(angle)),
                            (np.sin(angle), np.cos(angle)) ))

            rotated_points = []
            # print(rotated_points)
            for index, point in enumerate(remapped_points):
                # print(f'index: {index}')
                # print(f'point: {point}')
                newpoint = R @ point
                # print(newpoint)
                rotated_points.append(newpoint)
            rotated_points = np.array(rotated_points)
            
            # print(rotated_points)

            # For some reason this doesn't work - it fails to rotate the points by the correct angle. It has to be in a loop.
            # rotated_points = R @ remapped_points

            # Create plot
            # fig = plt.figure(figsize=(8, 8))
            # ax = fig.add_subplot(111)

            # Draw the points and the current simplex that is being tested
            # plt.scatter(x=remapped_points[:, 0], y=remapped_points[:, 1])
            # plt.plot(remapped_points[simplex, 0], remapped_points[simplex, 1], '#444444', linewidth=4)
            # plt.scatter(x=rotated_points[:, 0], y=rotated_points[:, 1])
            # plt.plot(rotated_points[simplex, 0], rotated_points[simplex, 1], 'k-', linewidth=5)
            # print(rotated_points[simplex, 0], rotated_points[simplex, 1])

            # # Draw the convex hulls
            # for simplex in simplexes:
            #     plt.plot(remapped_points[simplex, 0], remapped_points[simplex, 1], '#888888')
            #     plt.plot(rotated_points[simplex, 0], rotated_points[simplex, 1], '#555555')


            # Find the cartesian extremities
            # print(rotated_points[:, 0])
            x_min = np.min(rotated_points[:, 0])
            # print(x_min)
            x_max = np.max(rotated_points[:, 0])
            # print(x_max)
            y_min = np.min(rotated_points[:, 1])
            # print(y_min)
            y_max = np.max(rotated_points[:, 1])
            # print(y_max)

            # Draw bounding box 
            # plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], '#994400')
            # plt.show()

            bounding_area = (x_max - x_min) * (y_max - y_min)
            # print(f'bounding area: {bounding_area}')

            # If current bounding rectangle is the smallest so far
            if smallest_bounding_area == None or bounding_area < smallest_bounding_area:
                smallest_bounding_area = bounding_area
                smallest_bounding_rectangle = (x_min, x_max, y_min, y_max)
                aspect_ratio = (x_max - x_min) / (y_max - y_min)
                smallest_bounding_width = min((x_max - x_min), (y_max - y_min))
                smallest_bounding_length = max((x_max - x_min), (y_max - y_min))
                if aspect_ratio < 1.0:
                    aspect_ratio = 1/aspect_ratio
            
            # Unrotate the points
            # RINVERSE = R.T
            # unrotated_points = []
            # # print(rotated_points)
            # for index, point in enumerate(rotated_points):
            #     # print(f'index: {index}')
            #     # print(f'point: {point}')
            #     newpoint = RINVERSE @ point
            #     # print(newpoint)
            #     unrotated_points.append(newpoint)
            # unrotated_points = np.array(unrotated_points)

        # Unrotate the bounding box vertices
        RINVERSE = R.T
        translated_rotated_bounding_rectangle_vertices = np.array(([x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]))
        translated_bounding_rectangle_vertices = []
        for index, point in enumerate(translated_rotated_bounding_rectangle_vertices):
            newpoint = RINVERSE @ point
            translated_bounding_rectangle_vertices.append(newpoint)
        translated_bounding_rectangle_vertices = np.array(translated_bounding_rectangle_vertices)

            
        # Create plot
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111)
        # # plt.scatter(x=points[:, 0], y=points[:, 1])
        # ax.plot(np.append(translated_rotated_bounding_rectangle_vertices[:, 0], translated_rotated_bounding_rectangle_vertices[0, 0]), np.append(translated_rotated_bounding_rectangle_vertices[:, 1], translated_rotated_bounding_rectangle_vertices[0, 1]), '#994400', label='rotated')
        # ax.plot(np.append(translated_bounding_rectangle_vertices[:, 0], translated_bounding_rectangle_vertices[0, 0]), np.append(translated_bounding_rectangle_vertices[:, 1], translated_bounding_rectangle_vertices[0, 1]), '#004499', label='unrotated')
        # ax.scatter(x=remapped_points[:, 0], y=remapped_points[:, 1], color='#004499', label='translated')
        # # ax.scatter(x=unrotated_points[:, 0], y=unrotated_points[:, 1], label='unrotated')
        # ax.scatter(x=rotated_points[:, 0], y=rotated_points[:, 1], label='rotated')
        # ax.legend()
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        bounding_rectangle_vertices = translated_bounding_rectangle_vertices + centroid
        ax.plot(np.append(bounding_rectangle_vertices[:, 0], bounding_rectangle_vertices[0, 0]), np.append(bounding_rectangle_vertices[:, 1], bounding_rectangle_vertices[0, 1]), '#994400', label='unrotated')
        ax.scatter(x=edges[:, 0], y=edges[:, 1], label='original points')
        ax.set_aspect(1)
        ax.legend()
        plt.savefig(path / 'minimum_bbox.png')
        plt.close()
        
        # print(f'smallest bounding rectangle area: {smallest_bounding_area}')
        # print(f'smallest bounding rectangle: {smallest_bounding_rectangle}')
        # print(f'aspect ratio: {aspect_ratio}')
        # print(f'smallest bounding width: {smallest_bounding_width}')
        # print(f'smallest bounding length: {smallest_bounding_length}')



        return smallest_bounding_width, smallest_bounding_length, aspect_ratio



        
        

        

    

