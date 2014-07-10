// ==============================================================
// MitoThinning: This routine loads an ImageData binary volume
// and apply a thinning process over it, resulting in a new but
// topologically equivalent image.
// ==============================================================

#include <list>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <vtkMath.h>
#include <vtkPolyLine.h>
#include <vtkCellArray.h>
#include <vtkLongArray.h>
#include <vtkImageData.h>
#include <vtkDataArray.h>
#include <vtkTransform.h>
#include <vtkDataObject.h>
#include <vtkFloatArray.h>
#include <vtkVectorText.h>
#include <vtkInformation.h>
#include <vtkSphereSource.h>
#include <vtkSmartPointer.h>
#include <vtkStructuredPoints.h>
#include <vtkInformationVector.h>
#include <vtkUnsignedCharArray.h>
#include <vtkUnsignedShortArray.h>
#include <vtkStructuredPointsWriter.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkImageExtractComponents.h>
#include <vtkStructuredPointsReader.h>
#include <vtkImageGaussianSmooth.h>
#include <vtkAppendPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkContourFilter.h>
#include <vtkDoubleArray.h>
#include <vtkTIFFReader.h>
#include <vtkTIFFWriter.h>
#include <vtkPointData.h>
#include <vtkImageRFFT.h>
#include <vtkImageCast.h>
#include <vtkPoints.h>

#include "ssThinning.h"

    long int _small_cc_size = 0;
    bool _thinning_only = false;
    bool _export_graph_files = true;

    int ssdx[26] = {-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1,-1, 0, 1};
    int ssdy[26] = {-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int ssdz[26] = { 1, 1, 1, 0, 0, 0,-1,-1,-1, 1, 1, 1, 0, 0,-1,-1,-1, 1, 1, 1, 0, 0, 0,-1,-1,-1};

// Routine used to save an ImageData
void SaveImageData(vtkImageData *Image, const char FileName[]);

// Routine used to save a PolyData
void SavePolyData(vtkPolyData *PolyData, const char FileName[]);

// Routine used to save .gnet and .coo files representing
// the skeleton of the mitochondrial network.
void ExportGraphFiles(vtkPolyData *PolyData, long int nnodes, const char Prefix[]);

// Routine to create the file _nodes.vtk. This file contains
// the nodes represented as points in 3D space.
void ExportNodes(vtkPolyData *PolyData, const char Prefix[]);

// Calculate the length of a given edge.
double GetEdgeLength(vtkIdType edge, vtkPolyData *PolyData);

// Routine to delete all voxels located at boundaries of the
// image volume. Otherwise the algorithm will have problems
// checking the 3D neighborhood of those voxels.
void CleanImageBoundaries(vtkImageData *ImageData);

// Filter ImageData volume based on the size of connected components.
int FilterConnectedComponents(vtkImageData *ImageData);

// Routine used to check whether a voxel is locates at the border
// of the binary image or not. A border voxel is defined as those
// that have at least one of their 6-neighbors equal to zero.
bool IsBorder(vtkIdType id, vtkImageData *Image);

// Routine used to smooth the parametric curves that describe the
// skeleton edges. A simple average filter is used to do the job.
// Sigma represents the number of times the filter is applied.
void SmoothEdgesCoordinates(vtkPolyData *PolyData, double sigma);

// Returns the number of voxels around the voxel (x,y,z) with
// value given different of "value".
char GetNumberOfNeighborsWithoutValue(vtkImageData *Image, int x, int y, int z, long int value);

// Returns the number of voxels around the voxel (x,y,z) with
// value different of "value" in the vector "Volume".
char GetNumberOfNeighborsWithoutValue(vtkImageData *Image, vtkLongArray *Volume, int x, int y, int z, long int value);

// Returns the number of voxels around the voxel (x,y,z) with
// value given by "value".
char GetNumberOfNeighborsWithValue(vtkImageData *Image, int x, int y, int z, long int value);

// Returns the number of voxels around the voxel (x,y,z) with
// value "value" in the vector "Volume".
char GetNumberOfNeighborsWithValue(vtkImageData *Image, vtkLongArray *Volume, int x, int y, int z, long int value);

// Returns one neighbor of (x,y,z) with value "value" in the
// vector "Volume".
vtkIdType GetOneNeighborWithValue(int x, int y, int z, vtkImageData *Image, vtkLongArray *Volume, long int value);

// Returns one neighbor of (x,y,z) with value different of
// "value" in the vector "Volume".
vtkIdType GetOneNeighborWithoutValue(int x, int y, int z, vtkImageData *Image, vtkLongArray *Volume, long int value);

// Merge together junctions that touch each other.
bool JunctionsMerge(std::list<vtkIdType> Junctions, vtkImageData *Image, vtkLongArray *Volume);

// Track an edge starting at voxel (x,y,z) in the volume "Volume".
std::list<vtkIdType> GetEdgeStartingAt(int x, int y, int z, vtkImageData *Image, vtkLongArray *Volume);

// Returns an adjacency edge "edge_label".
long int GetOneAdjacentEdge(vtkPolyData *PolyData, long int edge_label, long int junction_label, bool *found_on_left);

// Track all the nodes and edges of a 3D structured thinned by
// the routine Thinning3D.
// @@FIX ME: Have to deal with degree-0 junctions (isolated voxels)
//           and have to implement the edge extension routine.
int Skeletonization(vtkImageData *Image, const char FileName[]);

// Replace two edges A and B attached to a node of degree 2
// with one edge C that corresponds to A + B. The degree of the
// node is set to -1 and A and B are moreved from PolyData.
bool MergeEdgesOfDegree2Nodes(vtkPolyData *PolyData, int *K);


/* ================================================================
   I/O ROUTINES
=================================================================*/

void SaveImageData(vtkImageData *Image, const char FileName[]) {
    #ifdef DEBUG
        printf("Saving ImageData File...\n");
    #endif

    vtkSmartPointer<vtkStructuredPointsWriter> writer = vtkSmartPointer<vtkStructuredPointsWriter>::New();
    writer -> SetInputData(Image);
    writer -> SetFileType(VTK_BINARY);
    writer -> SetFileName(FileName);
    writer -> Write();

    #ifdef DEBUG
        printf("\tFile Saved!\n");
    #endif
}

void SavePolyData(vtkPolyData *PolyData, const char FileName[]) {

    #ifdef DEBUG
        printf("Saving PolyData from XYZ list...\n");
    #endif

    #ifdef DEBUG
        printf("\t#Points in PolyData file: %llu.\n",(vtkIdType)PolyData->GetNumberOfPoints());
    #endif

    vtkSmartPointer<vtkPolyDataWriter> Writer = vtkSmartPointer<vtkPolyDataWriter>::New();
    Writer -> SetFileType(VTK_BINARY);
    Writer -> SetFileName(FileName);
    Writer -> SetInputData(PolyData);
    Writer -> Write();

    #ifdef DEBUG
        printf("\tFile Saved!\n");
    #endif
}

void ExportGraphFiles(vtkPolyData *PolyData, long int nnodes, long int *ValidId, const char Prefix[]) {
    
    #ifdef DEBUG
        printf("Saving .coo file...\n");
    #endif

    char _fullpath[256];

    vtkPoints *Points = PolyData -> GetPoints();

    double r[3];
    long int node, exact_nnodes = 0;
    for (node = 0; node < nnodes; node++) {
        if (ValidId[node]>=0) exact_nnodes++;
    }

    sprintf(_fullpath,"%s_graph.coo",Prefix);
    FILE *fcoo = fopen(_fullpath,"w");
    for (node = 0; node < nnodes; node++) {
        if (ValidId[node]>=0) {
            Points -> GetPoint(node,r);
            fprintf(fcoo,"%1.4f\t%1.4f\t%1.4f\n",r[0],r[1],r[2]);
        }
    }
    fclose(fcoo);

    double length;
    vtkIdType edge, npoints, i, j;
    sprintf(_fullpath,"%s_graph.gnet",Prefix);
    FILE *fgnet = fopen(_fullpath,"w");
    fprintf(fgnet,"%ld\n",exact_nnodes);
    for (edge = 0; edge < PolyData -> GetNumberOfCells(); edge++) {
        npoints = PolyData -> GetCell(edge) -> GetNumberOfPoints();
        i = PolyData -> GetCell(edge) -> GetPointId(0);
        j = PolyData -> GetCell(edge) -> GetPointId(npoints-1);
        if ( ValidId[i] >= 0 && ValidId[j] >= 0 ) {
            length = GetEdgeLength(edge,PolyData);
            fprintf(fgnet,"%ld\t%ld\t%1.5f\n",ValidId[i],ValidId[j],length);
        }
    }
    fclose(fgnet);

}

void ExportNodes(vtkPolyData *PolyData, const char Prefix[]) {
    
    #ifdef DEBUG
        printf("Exporting nodes...\n");
    #endif

    vtkSmartPointer<vtkPolyData> Nodes = vtkSmartPointer<vtkPolyData>::New();
    Nodes -> SetPoints(PolyData->GetPoints());

    char _fullpath[256];
    sprintf(_fullpath,"%s_nodes.vtk",Prefix);
    
    SavePolyData(Nodes,_fullpath);
}

double GetEdgeLength(vtkIdType edge, vtkPolyData *PolyData) {
    double r1[3], r2[3];
    double length = 0.0;
    for (vtkIdType n = 1; n < PolyData->GetCell(edge)->GetNumberOfPoints(); n++) {
        PolyData -> GetPoint(PolyData->GetCell(edge)->GetPointId(n-1),r1);
        PolyData -> GetPoint(PolyData->GetCell(edge)->GetPointId(n  ),r2);
        length += sqrt(pow(r2[0]-r1[0],2)+pow(r2[1]-r1[1],2)+pow(r2[2]-r1[2],2));
    }
    return length;
}

/* ================================================================
   IMAGE TRANSFORMATION
=================================================================*/

void CleanImageBoundaries(vtkImageData *ImageData) {
    #ifdef DEBUG
        printf("Cleaning the image boundaries...\n");
    #endif
    int p, q;
    int *Dim = ImageData -> GetDimensions();
    for ( p = Dim[0]; p--; ) {
        for ( q = Dim[1]; q--; ) {
            ImageData -> SetScalarComponentFromDouble(p,q,0,0,0);
            ImageData -> SetScalarComponentFromDouble(p,q,Dim[2]-1,0,0);
        }
    }
    for ( p = Dim[0]; p--; ) {
        for ( q = Dim[2]; q--; ) {
            ImageData -> SetScalarComponentFromDouble(p,0,q,0,0);
            ImageData -> SetScalarComponentFromDouble(p,Dim[1]-1,q,0,0);
        }
    }
    for ( p = Dim[1]; p--; ) {
        for ( q = Dim[2]; q--; ) {
            ImageData -> SetScalarComponentFromDouble(0,p,q,0,0);
            ImageData -> SetScalarComponentFromDouble(Dim[0]-1,p,q,0,0);
        }
    }
}

int FilterConnectedComponents(vtkImageData *ImageData) {

    #ifdef DEBUG
        printf("Filtering connected components...\n");
    #endif

    vtkIdType N = ImageData -> GetNumberOfPoints();
    int *Dim = ImageData -> GetDimensions();

    int cc_neigh = 26;
    vtkIdType i, s, ido, id;

    int x, y, z;
    double v, r[3];
    bool find = true;
    long long int ro[3];
    long int scluster, label;
    ro[0] = Dim[0] * Dim[1] * Dim[2];
    ro[1] = Dim[0] * Dim[1];

    #ifdef DEBUG
        printf("\tAllocating long int volume...\n");
    #endif

    vtkSmartPointer<vtkIdList> CurrA = vtkSmartPointer<vtkIdList>::New();
    vtkSmartPointer<vtkIdList> NextA = vtkSmartPointer<vtkIdList>::New();
    vtkSmartPointer<vtkLongArray> CSz = vtkSmartPointer<vtkLongArray>::New();
    vtkSmartPointer<vtkLongArray> Volume = vtkSmartPointer<vtkLongArray>::New();
    Volume -> SetNumberOfValues(N);
    Volume -> CopyComponent(0,ImageData->GetPointData()->GetScalars(),0);
    Volume -> Modified();

    #ifdef DEBUG
        printf("\tVolume allocated.\n");
    #endif

    label = 0;
    while (find) {
        for (s = 0; s < CurrA->GetNumberOfIds(); s++) {
            ido = CurrA -> GetId(s);
            ImageData -> GetPoint(ido,r);
            x = (int)r[0]; y = (int)r[1]; z = (int)r[2];
            for (i=cc_neigh; i--;) {
                id = ImageData -> FindPoint(x+ssdx[i],y+ssdy[i],z+ssdz[i]);
                v = Volume -> GetTuple1(id);
                if ((long int)v > 0) {
                    NextA -> InsertNextId(id);
                    Volume -> SetTuple1(id,-label);
                    scluster++;
                }
            }
        }
        if (!NextA->GetNumberOfIds()) {
            find = false;
            for (i=ro[0]; i--;) {
                z = (int)(i / ro[1]);
                y = (int)((i - (long long int)z * ro[1]) / ((long long int)Dim[0]));
                x = (int)(i - (long long )z * ro[1] - (long long int)y * Dim[0]);
                id = ImageData -> FindPoint(x,y,z);
                v = Volume -> GetTuple1(id);
                if ((long int)v > 0) {
                    find = true;
                    ro[0] = i;
                    break;
                }
                #ifdef DEBUG
                    if (!(i%(N/100))) {
                        printf("\t%1.1f\r",100-(100.0*i)/(N-1));
                    }
                #endif
            }
            if (label) {
                CSz -> InsertNextTuple1(scluster);
            }
            if (find) {
                label++;
                scluster = 1;
                Volume -> SetTuple1(id,-label);
                CurrA -> InsertNextId(id);
            }
        } else {
            CurrA -> Reset();
            CurrA -> DeepCopy(NextA);
            NextA -> Reset();
        }
    }

    #ifdef DEBUG
        printf("\n\tNumber of detected components: %ld\n",(long int)CSz->GetNumberOfTuples());
    #endif

    long int volume = 0;
    long int largecomp = 0;
    int nclusters = CSz -> GetNumberOfTuples();
    for (i = 0; i < CSz->GetNumberOfTuples(); i++) {
        largecomp = (  (long int)CSz->GetTuple1(i) > largecomp) ? (long int)CSz->GetTuple1(i) : largecomp;
        volume += (long int)CSz -> GetTuple1(i);
    }

    #ifdef DEBUG
        printf("\tSize of largest component: %ld\n",largecomp);
    #endif

    if (_small_cc_size<0) {
        #ifdef DEBUG
            printf("\tFiltering mode: Largest CC\n");
        #endif
        _small_cc_size = largecomp;
    } else {
        #ifdef DEBUG
            printf("\tFiltering mode: Minimum size = %ld\n",_small_cc_size);
        #endif

    }

    for (id=ImageData->GetNumberOfPoints(); id--;) {
        label = -(long int)Volume->GetTuple1(id);
        if (label) {
            if ((long int)CSz->GetTuple1(label-1) < _small_cc_size) {
                ImageData ->  GetPointData() -> GetScalars() -> SetTuple1(id,0);
            }
        }
    }

    ImageData -> GetPointData() -> GetScalars() -> Modified();

    #ifdef DEBUG
        printf("\tFiltering done!\n");
    #endif

    return 1;
}

/* ================================================================
   AUXILIAR ROUTINES
=================================================================*/


bool IsBorder(vtkIdType id, vtkImageData *Image) {
    double r[3];
    int x, y, z;
    Image -> GetPoint(id,r);
    x = (int)r[0]; y = (int)r[1]; z = (int)r[2];
    if(!Image->GetScalarComponentAsDouble(x,y,z,0))   return false;
    if(!Image->GetScalarComponentAsDouble(x+1,y,z,0)) return true;
    if(!Image->GetScalarComponentAsDouble(x-1,y,z,0)) return true;
    if(!Image->GetScalarComponentAsDouble(x,y+1,z,0)) return true;
    if(!Image->GetScalarComponentAsDouble(x,y-1,z,0)) return true;
    if(!Image->GetScalarComponentAsDouble(x,y,z+1,0)) return true;
    if(!Image->GetScalarComponentAsDouble(x,y,z-1,0)) return true;
    return false;
}

void SmoothEdgesCoordinates(vtkPolyData *PolyData, double sigma){

    double r1[3], r2[3], r3[3];
    long int id, line, nlines, n, s;
    vtkPoints *Points = PolyData -> GetPoints();

    vtkCell *Line;
    nlines = PolyData -> GetNumberOfCells();

    for (line = 0; line < nlines; line++){
        Line = PolyData -> GetCell(line);
        n = Line -> GetNumberOfPoints();
        double *X = new double[n];
        double *Y = new double[n];
        double *Z = new double[n];
        for (s = 0; s < int(sigma); s++) {
            for (id = 1; id < n-1; id++ ) {
                Points -> GetPoint(Line->GetPointId(id-1),r1);
                Points -> GetPoint(Line->GetPointId(id+0),r2);
                Points -> GetPoint(Line->GetPointId(id+1),r3);
                X[id] = 0.125*(r1[0]+6*r2[0]+r3[0]);
                Y[id] = 0.125*(r1[1]+6*r2[1]+r3[1]);
                Z[id] = (1.0/3.0)*(r1[2]+r2[2]+r3[2]);
            }
            for (id = 1; id < n-1; id++ ) {
                Points -> SetPoint(Line->GetPointId(id),X[id],Y[id],Z[id]);
            }
        }
        delete[] X; delete[] Y; delete[] Z;
    }
    PolyData -> Modified();
}

/* ================================================================
   THINNING 3D
=================================================================*/

int Thinning3D(const char FileName[]) {

    #ifdef DEBUG
        printf("Loading ImageData file...\n");
    #endif

    // Loading multi-paged TIFF file (Supported by VTK 6.2 and higher)
    char _fullpath[256];
    sprintf(_fullpath,"%s.tif",FileName);
    vtkSmartPointer<vtkTIFFReader> TIFFReader = vtkSmartPointer<vtkTIFFReader>::New();
    int errlog = TIFFReader -> CanReadFile(_fullpath);
    // File cannot be opened
    if (!errlog) {
        printf("File %s connot be opened.\n",_fullpath);
        return -1;
    }
    TIFFReader -> SetFileName(_fullpath);
    TIFFReader -> Update();

    vtkImageData *ImageData = TIFFReader -> GetOutput();
    vtkIdType N = ImageData -> GetNumberOfPoints();

    #ifdef DEBUG
        int *Dim = ImageData -> GetDimensions();
        printf("File: %s loaded.\n",_fullpath);
        printf("Dimensions = %d x %d x %d\n",Dim[0],Dim[1],Dim[2]);
        if (_thinning_only) printf("Thinning only active.\n");
    #endif

    CleanImageBoundaries(ImageData);

    if (_small_cc_size) {
        FilterConnectedComponents(ImageData);
    }

    int x, y, z;
    double r[3], v, vl;
    vtkIdType ndels, id;
    ssThinVox *STV = new ssThinVox();

    int i, j, ***Vol = new int**[3];
    for (i = 0; i < 3; i++) {
        Vol[i] = new int*[3];
            for (j = 0; j < 3; j++) {
                Vol[i][j] = new int[3];
            }
    }

    #ifdef DEBUG
        printf("Starting thinning process...\n");
    #endif

    std::list<vtkIdType> ToBeDeleted;
    std::list<vtkIdType> OnTheSurface;
    std::list<vtkIdType>::iterator itId;

    do {
        ndels = 0;
        for (id = N; id--;) {
            if (IsBorder(id, ImageData)) {
                OnTheSurface.insert(OnTheSurface.begin(),id);
            }
        }

        for (int direction = 6; direction--;) {
            for ( itId=OnTheSurface.begin(); itId!=OnTheSurface.end(); itId++) {
                id = *itId;
                ImageData -> GetPoint(id,r);
                x = (int)r[0]; y = (int)r[1]; z = (int)r[2];
                v = ImageData -> GetScalarComponentAsDouble(x,y,z,0);
                if (v) {
                    for (i = 0; i < 26; i++) {
                        vl = ImageData -> GetScalarComponentAsDouble(x+ssdx[i],y+ssdy[i],z+ssdz[i],0);
                        Vol[1+ssdx[i]][1+ssdy[i]][1+ssdz[i]] = (vl) ? 1 : 0;
                    }
                    if ( STV -> match(direction,Vol) ) ToBeDeleted.insert(ToBeDeleted.begin(),id);
                }
            }
            itId = ToBeDeleted.begin();
            while (itId != ToBeDeleted.end()) {
                ndels++;
                ImageData -> GetPoint(*itId,r);
                x = (int)r[0]; y = (int)r[1]; z = (int)r[2];
                ImageData -> SetScalarComponentFromDouble(x,y,z,0,0);
                ToBeDeleted.erase(itId++);
            }
        }

        #ifdef DEBUG
            printf("\t#Surface = %llu / #Deletions = %llu\n",(long long int)OnTheSurface.size(),ndels);
        #endif

        OnTheSurface.clear();

    } while(ndels);
    
    delete STV;

    #ifdef DEBUG
        printf("Thinning done!\n");
    #endif

    OnTheSurface.clear();
    ToBeDeleted.clear();

    if (_thinning_only) {
        char _fullpath[256];
        sprintf(_fullpath,"%s_thinning.vtk",FileName);
        SaveImageData(ImageData,_fullpath);
        return 1;
    }

    return Skeletonization(ImageData,FileName);
}

/* ================================================================
   SKELETONIZATION
=================================================================*/

char GetNumberOfNeighborsWithoutValue(vtkImageData *Image, int x, int y, int z, long int value) {
    double r[3];
    char nn = 0;
    for (char k = 26; k--;) {
        if (Image->GetScalarComponentAsDouble(x+ssdx[k],y+ssdy[k],z+ssdz[k],0) != value) nn++;
    }
    return nn;
}

char GetNumberOfNeighborsWithoutValue(vtkImageData *Image, vtkLongArray *Volume, int x, int y, int z, long int value) {
    double r[3];
    char nn = 0;
    vtkIdType idk;
    for (char k = 26; k--;) {
        idk = Image->FindPoint(x+ssdx[k],y+ssdy[k],z+ssdz[k]);
        if (Volume->GetTuple1(idk) != value) nn++;
    }
    return nn;
}

char GetNumberOfNeighborsWithValue(vtkImageData *Image, int x, int y, int z, long int value) {
    double r[3];
    char nn = 0;
    for (char k = 26; k--;) {
        if (Image->GetScalarComponentAsDouble(x+ssdx[k],y+ssdy[k],z+ssdz[k],0) == value) nn++;
    }
    return nn;
}

char GetNumberOfNeighborsWithValue(vtkImageData *Image, vtkLongArray *Volume, int x, int y, int z, long int value) {
    double r[3];
    char nn = 0;
    vtkIdType idk;
    for (char k = 26; k--;) {
        idk = Image -> FindPoint(x+ssdx[k],y+ssdy[k],z+ssdz[k]);
        if (Volume->GetTuple1(idk) == value) nn++;
    }
    return nn;
}

vtkIdType GetOneNeighborWithValue(int x, int y, int z, vtkImageData *Image, vtkLongArray *Volume, long int value) {
    vtkIdType idk = 0; // We can do it because, by construction, the voxel at id 0 should always be empty
    for (char k = 26; k--;) {
        idk = Image -> FindPoint(x+ssdx[k],y+ssdy[k],z+ssdz[k]);
        if (Volume -> GetTuple1(idk) == value) {
            return idk;
        }
    }
    return 0;
}

vtkIdType GetOneNeighborWithoutValue(int x, int y, int z, vtkImageData *Image, vtkLongArray *Volume, long int value) {
    vtkIdType idk = 0; // We can do it because, by construction, the voxel at id 0 should always be empty
    for (char k = 26; k--;) {
        idk = Image -> FindPoint(x+ssdx[k],y+ssdy[k],z+ssdz[k]);
        if (Volume -> GetTuple1(idk) && Volume -> GetTuple1(idk) != value) {
            return idk;
        }
    }
    return 0;
}

bool JunctionsMerge(std::list<vtkIdType> Junctions, vtkImageData *Image, vtkLongArray *Volume) {

    char nk;
    double r[3];
    vtkIdType id;
    int x, y, z, k;
    bool _has_changed = false;
    std::list<vtkIdType>::iterator itId;
    long int junction_label, neigh_junction_label;
    for (itId=Junctions.begin(); itId!=Junctions.end(); itId++) {
        Image -> GetPoint(*itId,r);
        junction_label = Volume -> GetTuple1(*itId);
        x = (int)r[0]; y = (int)r[1]; z = (int)r[2];
        for (k = 0; k < 26; k++) {
            id = Image->FindPoint(x+ssdx[k],y+ssdy[k],z+ssdz[k]);
            neigh_junction_label = Volume -> GetTuple1(id);
            if (junction_label < neigh_junction_label) {
                _has_changed = true;
                Volume -> SetTuple1(id,junction_label);
            }
        }
    }
    Volume -> Modified();
    return _has_changed;
}

std::list<vtkIdType> GetEdgeStartingAt(int x, int y, int z, vtkImageData *Image, vtkLongArray *Volume) {
    double r[3];
    vtkIdType idk;
    std::list<vtkIdType> Edge;
    do {
        idk = GetOneNeighborWithValue(x,y,z,Image,Volume,-1);
        if (idk) {
            Volume -> SetTuple1(idk,0);
            Edge.insert(Edge.end(),idk);
            Image -> GetPoint(idk,r);
            x = (int)r[0]; y = (int)r[1]; z = (int)r[2];
        }
    } while(idk);
    return Edge;
}

long int GetOneAdjacentEdge(vtkPolyData *PolyData, long int edge_label, long int original_source, bool *_common_source) {
    vtkCell *Edge;
    long int source, target;
    for (long int edge = PolyData->GetNumberOfCells();edge--;) {
        if (edge != edge_label) {
            Edge = PolyData -> GetCell((vtkIdType)edge);
            source = Edge -> GetPointId(0);
            target = Edge -> GetPointId(Edge->GetNumberOfPoints()-1);
            if (original_source==source) {
                *_common_source = true;
                return edge;
            }
            if (original_source==target) {
                *_common_source = false;
                return edge;
            }
        }
    }
    return -1;
}


int Skeletonization(vtkImageData *Image, const char FileName[]) {

    #ifdef DEBUG
        printf("Starting skeletonization process...\n");
    #endif

    double r[3];
    int x, y, z;
    vtkIdType id;
    long int junction_label = 1;
    vtkIdType N = Image -> GetNumberOfPoints();

    std::list<vtkIdType> Junctions;
    std::list<vtkIdType>::iterator itId;

    #ifdef DEBUG
        printf("\tAllocating long int volume...\n");
    #endif

    vtkSmartPointer<vtkLongArray> Volume = vtkSmartPointer<vtkLongArray>::New();
    Volume -> SetNumberOfComponents(0);
    Volume -> SetNumberOfTuples(N);
    for (id = N; id--;) {
        if (Image -> GetPointData() -> GetScalars() -> GetTuple1(id)) {
            Volume -> SetTuple1(id,-1);
        } else {
            Volume -> SetTuple1(id,0);
        }
    }
    Volume -> Modified();

    #ifdef DEBUG
        printf("\tLong int volume allocated...\n");
        printf("\tDetecting junctions...\n");
    #endif

    for (id = N; id--;) {
        Image -> GetPoint(id,r);
        x = (int)r[0]; y = (int)r[1]; z = (int)r[2];
        if (Image->GetScalarComponentAsDouble(x,y,z,0)) {
            if (GetNumberOfNeighborsWithoutValue(Image,x,y,z,0) != 2) {
                Junctions.insert(Junctions.begin(),id);
                Volume -> SetTuple1(id,junction_label);
                junction_label++;
            }
        }
    }
    Volume -> Modified();

    if (!Junctions.size()) {
        printf("Z-stack seems to be empty. Aborting...\n");
        return 0;
    }

    #ifdef DEBUG
        printf("\t#Junctions before merging = %ld\n",Junctions.size());
    #endif

    // MERGING: During the merging process, voxels belonging to
    // the same junction are merger together forming nodes.
    while (JunctionsMerge(Junctions,Image,Volume));

    vtkIdType NumberOfVoxelsOnEdges = 0;
    for (id = N; id--;) {
        if (Volume->GetTuple1(id)==-1) NumberOfVoxelsOnEdges++;
    }

    // Listing all nodes label we have until this point
    std::list<long int> Labels;
    std::list<long int>::iterator itLabel;
    for (itId=Junctions.begin(); itId!=Junctions.end(); itId++) {
        Labels.insert(Labels.begin(),(long int)Volume->GetTuple1(*itId));
    }

    // UNIQUE of labels
    Labels.sort(); // must sort first
    Labels.unique();
    long int NumberOfNodes = Labels.size();

    #ifdef DEBUG
        printf("\t#Junctions (nodes) after merging = %ld\n",NumberOfNodes);
    #endif

    // COORDINATES of nodes
    // Vectors with static size to make the average calculation easy.
    long int node;
    double *X = new double[NumberOfNodes]; //Vector for x-coordinate
    double *Y = new double[NumberOfNodes]; //Vector for y-coordinate
    double *Z = new double[NumberOfNodes]; //Vector for z-coordinate
    double *S = new double[NumberOfNodes]; //Vector for junctions size
       int *K = new int[NumberOfNodes];    //Vector for junctions degree
    for (node = 0; node < NumberOfNodes; node++) {
        K[node] = 0;
        X[node] = Y[node] = Z[node] = S[node] = 0.0;
    }

    for (itId=Junctions.begin(); itId!=Junctions.end(); itId++) {
        junction_label = (long int)Volume -> GetTuple1(*itId);
        itLabel = std::find(Labels.begin(),Labels.end(),junction_label);
        node = (long int)std::distance(Labels.begin(),itLabel);
        Volume -> SetTuple1(*itId,node+1);
        Image -> GetPoint(*itId,r);
        X[node] += r[0];
        Y[node] += r[1];
        Z[node] += r[2];
        S[node] ++;
    }
    Volume -> Modified();

    // Dynamic list to make easy the insertion of the coordinates
    // of remaining voxels detected during the edge tracking process.
    std::list<double> PointsListX;
    std::list<double> PointsListY;
    std::list<double> PointsListZ;
    for (node=0;node<NumberOfNodes;node++) {
        PointsListX.insert(PointsListX.end(),X[node]/S[node]);
        PointsListY.insert(PointsListY.end(),Y[node]/S[node]);
        PointsListZ.insert(PointsListZ.end(),Z[node]/S[node]);
    }

    delete[] X; delete[] Y; delete[] Z; delete[] S;

    // CellArray to add new edges as they are being tracked.
    vtkSmartPointer<vtkCellArray> EdgeArray = vtkSmartPointer<vtkCellArray>::New();

    bool _should_add;
    itId = Junctions.begin();
    long int voxel_label = NumberOfNodes;
    long int junction_label_left, junction_label_right;
    long int source_node, target_node;
    while (itId != Junctions.end()) {
        Image -> GetPoint(*itId,r);
        x = (int)r[0]; y = (int)r[1]; z = (int)r[2];
        if (GetNumberOfNeighborsWithValue(Image,Volume,x,y,z,-1)) {
            _should_add = true;

            //Tracking new edge
            std::list<vtkIdType> Edge = GetEdgeStartingAt(x,y,z,Image,Volume);

            //Identifying junctions on left side of the edge
            source_node = Volume -> GetTuple1(*itId);

            //Identifying junctions on right side of the edge
            Image -> GetPoint(Edge.back(),r);
            x = (int)r[0]; y = (int)r[1]; z = (int)r[2];
            target_node = Volume -> GetTuple1(GetOneNeighborWithoutValue(x,y,z,Image,Volume,source_node));

            // When target_node is equal to 0 at this point, we are
            // dealing with loops. These loops can be real loops or noise
            // loops of very small length. This very short loops are often
            // related to noise around junctions or 1-voxel holes in the
            // data.
            // @@PARAMETER: Minimum loop length (in voxels)
            if (!target_node) {
                target_node = source_node;
                if (Edge.size()<3) _should_add = false;
            }

            if (_should_add) {
                // Creating new cell for this edge
                EdgeArray -> InsertNextCell(Edge.size()+2);
                // Adding node 1
                EdgeArray -> InsertCellPoint(source_node-1);
                // Addint the new points coordinates as well as the edge itself
                for (std::list<vtkIdType>::iterator itIde=Edge.begin(); itIde!=Edge.end(); itIde++) {
                    Image -> GetPoint(*itIde,r);
                    PointsListX.insert(PointsListX.end(),r[0]);
                    PointsListY.insert(PointsListY.end(),r[1]);
                    PointsListZ.insert(PointsListZ.end(),r[2]);
                    EdgeArray -> InsertCellPoint(voxel_label);
                    voxel_label++;
                }
                // Adding node 2
                EdgeArray -> InsertCellPoint(target_node-1);
                // Updating nodes degree
                K[source_node-1]++;
                K[target_node-1]++;
            }
            
        } else {
            Junctions.erase(itId++);
        }
    }

    vtkSmartPointer<vtkPoints> Points = vtkSmartPointer<vtkPoints>::New();
    Points -> SetNumberOfPoints(PointsListX.size());

    #ifdef DEBUG
        printf("#Points in vtkPoints = %lld\n",Points -> GetNumberOfPoints());
    #endif

    voxel_label = 0;
    std::list<double>::iterator itX = PointsListX.begin();
    std::list<double>::iterator itY = PointsListY.begin();
    std::list<double>::iterator itZ = PointsListZ.begin();
    while (itX != PointsListX.end()) {
        Points -> SetPoint(voxel_label,*itX,*itY,*itZ);
        itX++; itY++; itZ++;
        voxel_label++;
    }
    Points -> Modified();

    PointsListX.clear();
    PointsListY.clear();
    PointsListZ.clear();

    // Creating raw polyData
    vtkPolyData *PolyData = vtkPolyData::New();
    PolyData -> SetPoints(Points);
    PolyData -> SetLines(EdgeArray);
    PolyData -> Modified();
    PolyData -> BuildLinks();

    #ifdef DEBUG
        printf("\t#Edges before filtering = %lld\n",PolyData->GetNumberOfCells());
    #endif

    // PolyData filtering by removing degree-2 nodes. These nodes rise
    // for two reasons: 1) very short edges that are not detected,
    // although the bifurcation is detected. 2) Short loops that were
    // removed.
    while (MergeEdgesOfDegree2Nodes(PolyData,K));

    //Creating a list with final Ids of nodes after degree-2 nodes
    //deletetion.
    long int valid_id = 0;
    long int *ValidId = new long int[NumberOfNodes];
    for (node = 0; node < NumberOfNodes; node++) {
        if (K[node] > 0) {
            ValidId[node] = valid_id;
            valid_id++;
        } else {
            ValidId[node] = -1;
        }
    }

    #ifdef DEBUG
        printf("\t#Edges after filtering = %lld\n",PolyData->GetNumberOfCells());
        printf("Skeletonization done!\n");
    #endif

    SmoothEdgesCoordinates(PolyData,3);

    if (_export_graph_files) ExportGraphFiles(PolyData,NumberOfNodes,ValidId,FileName);

    ExportNodes(PolyData,FileName);

    char _fullpath[256];
    sprintf(_fullpath,"%s_skeleton.vtk",FileName);
    SavePolyData(PolyData,_fullpath);

    //@FIX ME: Expand edges and nodes with degree zero.

    return 1;

}

bool MergeEdgesOfDegree2Nodes(vtkPolyData *PolyData, int *K) {

    // Given this edge: (source) o---->----o (target), it's necessary
    // to check whether either source or target are nodes of degree 2.

    std::list<vtkIdType> Merg;
    std::list<vtkIdType>::iterator itId;

    vtkCell *Edge, *NeighEdge;
    long int i, source, target, edge, edge_length, neigh_edge, neigh_edge_length;
    bool _common_source, _merged = false;

    for (edge = 0; edge < PolyData -> GetNumberOfCells(); edge++) {

        Edge = PolyData -> GetCell(edge);
        edge_length = Edge -> GetNumberOfPoints();
        source = (long int)Edge -> GetPointId(0);
        target = (long int)Edge -> GetPointId(Edge->GetNumberOfPoints()-1);

        if (K[source]==2 && source!=target) {

            //  [  neigh  | original ]
            //  o----?----o---->-----o
            //            s          t

            _merged = true;
            neigh_edge = GetOneAdjacentEdge(PolyData,edge,source,&_common_source);
            NeighEdge = PolyData -> GetCell((vtkIdType)neigh_edge);
            neigh_edge_length = NeighEdge -> GetNumberOfPoints();

            if (_common_source) {//  o----<----o---->----o
                for (i=0;i<edge_length;i++)
                    Merg.insert(Merg.begin(),PolyData->GetCell(edge)->GetPointId(i));
                for (i=1;i<neigh_edge_length;i++)
                    Merg.insert(Merg.end(),PolyData->GetCell(neigh_edge)->GetPointId(i));
            } else { //  o---->----o---->----o
                for (i=0;i<neigh_edge_length;i++)
                    Merg.insert(Merg.end(),PolyData->GetCell(neigh_edge)->GetPointId(i));                
                for (i=1;i<edge_length;i++)
                    Merg.insert(Merg.end(),PolyData->GetCell(edge)->GetPointId(i));
            }
            K[source] = -1; // Tagged as not valid node

        } else if (K[target]==2 && source!=target) {

            //  [ original |  neigh  ]
            //  o---->-----o----?----o
            //  s          t

            _merged = true;
            neigh_edge = GetOneAdjacentEdge(PolyData,edge,target,&_common_source);
            NeighEdge = PolyData -> GetCell((vtkIdType)neigh_edge);
            neigh_edge_length = NeighEdge -> GetNumberOfPoints();

            if (_common_source) {//  o---->----o---->----o
                for (i=0;i<edge_length;i++)
                    Merg.insert(Merg.end(),PolyData->GetCell(edge)->GetPointId(i));
                for (i=1;i<neigh_edge_length;i++)
                    Merg.insert(Merg.end(),PolyData->GetCell(neigh_edge)->GetPointId(i));
            } else {//  o---->----o----<----o
                for (i=0;i<edge_length;i++)
                    Merg.insert(Merg.end(),PolyData->GetCell(edge)->GetPointId(i));
                for (i=neigh_edge_length-1;i--;)
                    Merg.insert(Merg.end(),PolyData->GetCell(neigh_edge)->GetPointId(i));                
            }
            K[target] = -1; // Tagged as not valid node
        }

        if (_merged) {
            i = 0;
            // Removing the two pieces that were merged together
            PolyData -> BuildLinks();
            PolyData -> DeleteCell(edge);
            PolyData -> DeleteCell(neigh_edge);
            PolyData -> RemoveDeletedCells();
            // Adding the new "merged" edge
            vtkIdType IdList[Merg.size()];
            for (itId=Merg.begin(); itId!=Merg.end(); itId++) {
                IdList[i] = *itId;
                i++;
            }
            PolyData -> InsertNextCell(VTK_POLY_LINE,Merg.size(),IdList);
            Merg.clear();
            return true;
        }

    }
    return false;
}

/* ================================================================
   MAIN ROUTINE
=================================================================*/

void _help() {
    printf("\n 3DSkell options:\n");
    printf("\t -path, -cc, -graph_off, -thinning_only\n");
}

int main(int argc, char *argv[]) {     

    int i;
    char _impath[128];
    long int _cc_value;
    sprintf(_impath,"");

    // Collecting input parameters
    for (i = 0; i < argc; i++) {
        if (!strcmp(argv[i],"-path")) {
            sprintf(_impath,"%s//",argv[i+1]);
        }
        if (!strcmp(argv[i],"-cc")) {
            _cc_value = (long int)atof(argv[i+1]);
        }
        if (!strcmp(argv[i],"-graph_off")) {
            _export_graph_files = false;
        }
        if (!strcmp(argv[i],"-thinning_only")) {
            _thinning_only = true;
        }
        if (!strcmp(argv[i],"-?")) {
            _help();
            return 1;
        }
    }

    // Generating list of files to run
    char _cmd[256];
    sprintf(_cmd,"ls %s*.tif | sed -e 's/.tif//' > %smitograph.files",_impath,_impath);
    system(_cmd);

    // Thinning
    char _tifffilename[256];
    char _tifflistpath[128];
    sprintf(_tifflistpath,"%smitograph.files",_impath);
    FILE *f = fopen(_tifflistpath,"r");
    while (fgets(_tifffilename,256, f) != NULL) {
        if (_cc_value==0) {
            _small_cc_size = 0;
        } else if (_cc_value==-1) {
            _small_cc_size = -1;
        }
        _tifffilename[strcspn(_tifffilename, "\n" )] = '\0';
        Thinning3D(_tifffilename);
    }
    fclose(f);

    return 1;
}
