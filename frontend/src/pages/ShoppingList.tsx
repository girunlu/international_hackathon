import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Layout from "@/components/Layout";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import { formatQuantityLabel } from "@/lib/utils";
import {
  addShoppingListItem,
  checkWasteHistory,
  detectItem,
  getShoppingList,
  markItemsAsBought,
  recordSavedItem,
  removeShoppingListItem,
  type ShoppingListEntry,
} from "@/lib/backend-api";
import { Camera, Loader2, Plus, ShoppingBag, Trash2, Upload } from "lucide-react";

interface PendingItemState {
  name: string;
  requestedQuantity: number;
  requestedUnit?: string | null;
  wastedAmountLabel?: string;
  suggestedQuantity?: number;
  suggestedUnit?: string | null;
  suggestedAmountLabel?: string;
}

const ShoppingList = () => {
  const [shoppingItems, setShoppingItems] = useState<ShoppingListEntry[]>([]);
  const [isListLoading, setIsListLoading] = useState<boolean>(true);
  const [isSubmittingItem, setIsSubmittingItem] = useState<boolean>(false);
  const [isMarkingBought, setIsMarkingBought] = useState<boolean>(false);
  const [newItemName, setNewItemName] = useState("");
  const [newItemAmount, setNewItemAmount] = useState("");
  const [newItemUnit, setNewItemUnit] = useState("");
  const [pendingItem, setPendingItem] = useState<PendingItemState | null>(null);
  const [cameraImage, setCameraImage] = useState<string | null>(null);
  const [galleryImage, setGalleryImage] = useState<string | null>(null);
  const [isDetectingItem, setIsDetectingItem] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isCameraModalOpen, setIsCameraModalOpen] = useState<boolean>(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const { toast } = useToast();

  const resetForm = useCallback(() => {
    setNewItemName("");
    setNewItemAmount("");
    setNewItemUnit("");
    setCameraImage(null);
    setGalleryImage(null);
  }, []);

  const loadShoppingList = useCallback(async () => {
    try {
      setIsListLoading(true);
      const items = await getShoppingList();
      setShoppingItems(items);
    } catch (error) {
      console.error("Error loading shopping list:", error);
      toast({
        title: "Unable to load shopping list",
        description: "Please make sure the backend is running and try again.",
        variant: "destructive",
      });
    } finally {
      setIsListLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    loadShoppingList();
  }, [loadShoppingList]);

  const performAdd = useCallback(
    async (itemName: string, quantityNumeric: number, unit: string | null, toastMessage?: string) => {
      const normalizedUnit = unit ?? null;
      const quantityLabel = formatQuantityLabel(quantityNumeric, normalizedUnit ?? undefined);

      await addShoppingListItem(itemName, quantityNumeric, normalizedUnit);
      await loadShoppingList();

      toast({
        title: "Item added",
        description: toastMessage ?? `${itemName} (${quantityLabel}) has been added to your shopping list`,
      });
      resetForm();
    },
    [loadShoppingList, resetForm, toast]
  );

  const attemptAddItem = useCallback(
    async (itemName: string, quantityNumeric: number, unit?: string | null) => {
      const normalizedUnit = unit ?? null;
      try {
        const result = await checkWasteHistory(itemName, quantityNumeric, normalizedUnit);

        if (result?.hasWaste) {
          const wastedAmountLabel =
            result.wastedAmount ?? (result as { amount?: string }).amount;

          const suggestedQuantityRaw =
            typeof result.suggestedQuantityNumeric === "number"
              ? result.suggestedQuantityNumeric
              : (result as { suggested_quantity_numeric?: number }).suggested_quantity_numeric;

          const suggestedAmountRaw =
            result.suggestedAmount ??
            (result as { suggested_amount?: string }).suggested_amount ??
            (result as { suggestedAmount?: string }).suggestedAmount;

          const suggestedQuantity =
            typeof suggestedQuantityRaw === "number"
              ? suggestedQuantityRaw
              : suggestedAmountRaw
              ? parseFloat(suggestedAmountRaw)
              : undefined;

          const suggestedUnit =
            typeof result.suggestedUnit === "string"
              ? result.suggestedUnit
              : (result as { suggested_unit?: string }).suggested_unit ?? normalizedUnit;

          const suggestedAmountLabel =
            suggestedAmountRaw ??
            (typeof suggestedQuantity === "number"
              ? formatQuantityLabel(suggestedQuantity, suggestedUnit ?? undefined)
              : undefined);

          setPendingItem({
            name: itemName,
            requestedQuantity: quantityNumeric,
            requestedUnit: normalizedUnit,
            wastedAmountLabel,
            suggestedQuantity,
            suggestedUnit,
            suggestedAmountLabel,
          });
          return "requires-confirmation" as const;
        }

        await performAdd(itemName, quantityNumeric, normalizedUnit);
        return "added" as const;
      } catch (error) {
        console.error("Error adding shopping list item:", error);
        toast({
          title: "Failed to add item",
          description: "Please try again later.",
          variant: "destructive",
        });
        return "error" as const;
      }
    },
    [performAdd, toast]
  );

  const handleAdd = async () => {
    const trimmedName = newItemName.trim();
    if (!trimmedName || !newItemAmount) {
      toast({
        title: "Missing information",
        description: "Please provide an item name and quantity.",
        variant: "destructive",
      });
      return;
    }

    const quantityNumeric = parseFloat(newItemAmount);
    if (!Number.isFinite(quantityNumeric) || quantityNumeric <= 0) {
      toast({
        title: "Invalid amount",
        description: "Enter a positive numeric quantity.",
        variant: "destructive",
      });
      return;
    }

    const unit = newItemUnit.trim() || null;

    setIsSubmittingItem(true);
    const result = await attemptAddItem(trimmedName, quantityNumeric, unit);
    setIsSubmittingItem(false);

    if (result === "added") {
      resetForm();
    }
  };

  const handleConfirmAdd = async () => {
    if (!pendingItem) return;

    setIsSubmittingItem(true);
    try {
      await performAdd(
        pendingItem.name,
        pendingItem.requestedQuantity,
        pendingItem.requestedUnit ?? null,
        `${pendingItem.name} (${formatQuantityLabel(
          pendingItem.requestedQuantity,
          pendingItem.requestedUnit ?? undefined
        )}) has been added to your shopping list`
      );
      setPendingItem(null);
    } catch (error) {
      // performAdd already handles toasts on failure
    } finally {
      setIsSubmittingItem(false);
    }
  };

  const handleUseSuggestedAmount = async () => {
    if (!pendingItem) return;

    const suggestedQuantity =
      typeof pendingItem.suggestedQuantity === "number"
        ? pendingItem.suggestedQuantity
        : pendingItem.requestedQuantity;

    const suggestedUnit = pendingItem.suggestedUnit ?? pendingItem.requestedUnit ?? null;

    setIsSubmittingItem(true);
    try {
      const usingSuggestedReduction =
        typeof pendingItem.suggestedQuantity === "number" &&
        pendingItem.suggestedQuantity !== pendingItem.requestedQuantity;

      const formattedQuantityLabel = formatQuantityLabel(
        suggestedQuantity,
        suggestedUnit ?? undefined
      );

      if (usingSuggestedReduction) {
        await recordSavedItem({
          itemName: pendingItem.name,
          originalQuantity: pendingItem.requestedQuantity,
          originalUnit: pendingItem.requestedUnit ?? null,
          savedQuantity: pendingItem.suggestedQuantity,
          savedUnit: suggestedUnit,
        });
        await loadShoppingList();
        toast({
          title: "Item added",
          description: `${pendingItem.name} (${formattedQuantityLabel}) has been added to your shopping list`,
        });
        resetForm();
      } else {
        await performAdd(
          pendingItem.name,
          suggestedQuantity,
          suggestedUnit,
          `${pendingItem.name} (${formattedQuantityLabel}) added using the suggested amount`
        );
      }

      toast({
        title: "Smart choice!",
        description: `Adjusted quantity to ${formatQuantityLabel(
          suggestedQuantity,
          suggestedUnit ?? undefined
        )} to reduce waste.`,
      });

      setPendingItem(null);
    } catch (error) {
      // performAdd already handles toasts on failure
    } finally {
      setIsSubmittingItem(false);
    }
  };

  const handleCancelAdd = () => {
    setPendingItem(null);
    toast({
      title: "Cancelled",
      description: "Item was not added to your list",
    });
  };

  const handleDelete = async (id: number) => {
    try {
      const result = await removeShoppingListItem(id);
      await loadShoppingList();
      if (!result?.success) {
        toast({
          title: "Unable to confirm removal",
          description: "List refreshed, but backend did not confirm the delete.",
          variant: "destructive",
        });
        return;
      }
      toast({
        title: "Item removed",
        description: "Item has been removed from your shopping list",
      });
    } catch (error) {
      console.error("Error removing shopping list item:", error);
      toast({
        title: "Failed to remove item",
        description: "Please try again later.",
        variant: "destructive",
      });
    }
  };

  const handleMarkAllBought = async () => {
    if (shoppingItems.length === 0) {
      toast({
        title: "No items",
        description: "Your shopping list is empty",
        variant: "destructive",
      });
      return;
    }

    try {
      setIsMarkingBought(true);
      const itemsPayload = shoppingItems.map((item) => ({
        item_name: item.item_name,
        quantity_numeric: item.quantity_numeric,
        unit: item.quantity_unit ?? undefined,
      }));

      await markItemsAsBought(itemsPayload);
      await loadShoppingList();

      toast({
        title: "Shopping complete!",
        description: `${itemsPayload.length} items marked as bought`,
      });
    } catch (error) {
      console.error("Error marking items as bought:", error);
      toast({
        title: "Failed to mark items",
        description: "Please try again later.",
        variant: "destructive",
      });
    } finally {
      setIsMarkingBought(false);
    }
  };

  const stopCamera = useCallback(() => {
    const activeStream = cameraStreamRef.current;
    if (activeStream) {
      activeStream.getTracks().forEach((track) => track.stop());
    }
    cameraStreamRef.current = null;
    setCameraStream(null);
  }, []);

  const startCamera = useCallback(async () => {
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("Camera access is not supported on this device.");
      }

      setCameraError(null);
      setIsCameraModalOpen(true);
      stopCamera();

      const constraintsBase: MediaStreamConstraints = {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      };

      const tryGetStream = async (constraints: MediaStreamConstraints) => {
        try {
          return await navigator.mediaDevices.getUserMedia(constraints);
        } catch (error) {
          return null;
        }
      };

      let stream: MediaStream | null = null;

      stream = await tryGetStream({
        ...constraintsBase,
        video: {
          ...(constraintsBase.video as MediaTrackConstraints),
          facingMode: { ideal: "environment" },
        },
      });

      if (!stream) {
        stream = await tryGetStream({
          ...constraintsBase,
          video: {
            ...(constraintsBase.video as MediaTrackConstraints),
            facingMode: { ideal: "user" },
          },
        });
      }

      if (!stream) {
        stream = await tryGetStream(constraintsBase);
      }

      if (!stream) {
        throw new Error("Unable to access any camera device.");
      }

      cameraStreamRef.current = stream;
      setCameraStream(stream);
      setCameraImage(null);
    } catch (error) {
      console.error("Unable to start camera:", error);
      setCameraError(
        error instanceof Error ? error.message : "Camera access was denied or no camera is available."
      );
      toast({
        title: "Camera unavailable",
        description: "Please allow camera access, ensure a webcam is connected, or try uploading a photo instead.",
        variant: "destructive",
      });
    }
  }, [stopCamera, toast]);

  useEffect(() => {
    if (videoRef.current && cameraStream) {
      videoRef.current.srcObject = cameraStream;
      const attemptPlay = async () => {
        try {
          await videoRef.current?.play();
        } catch (error) {
          console.error("Unable to start video playback:", error);
        }
      };
      attemptPlay();

      const handleCanPlay = () => {
        attemptPlay();
      };

      const videoElement = videoRef.current;
      videoElement?.addEventListener("canplay", handleCanPlay, { once: true });

      return () => {
        videoElement?.removeEventListener("canplay", handleCanPlay);
      };
    }
  }, [cameraStream]);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  const handleCloseCameraModal = useCallback(() => {
    setIsCameraModalOpen(false);
    stopCamera();
    setCameraError(null);
  }, [stopCamera]);

  const captureCameraFrame = useCallback(async () => {
    const videoElement = videoRef.current;
    const canvasElement = canvasRef.current;
    if (!videoElement || !canvasElement) {
      return;
    }

    const width = videoElement.videoWidth;
    const height = videoElement.videoHeight;
    if (!width || !height) {
      return;
    }

    canvasElement.width = width;
    canvasElement.height = height;
    const context = canvasElement.getContext("2d");
    if (!context) {
      return;
    }

    context.drawImage(videoElement, 0, 0, width, height);
    const base64Image = canvasElement.toDataURL("image/jpeg");

    setCameraImage(base64Image);
    setIsCameraModalOpen(false);
    stopCamera();

    setIsDetectingItem(true);
    try {
      await detectItem(base64Image);
      // TODO: populate detection result once UI flow is defined
    } catch (error) {
      console.error("Error detecting item from camera image:", error);
      toast({
        title: "Detection failed",
        description: "Please add the item manually.",
        variant: "destructive",
      });
    } finally {
      setIsDetectingItem(false);
    }
  }, [stopCamera, toast]);

  const handleGalleryUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    const reader = new FileReader();
    reader.onloadend = async () => {
      const base64Image = reader.result as string;
      setGalleryImage(base64Image);
      setIsDetectingItem(true);

      try {
        await detectItem(base64Image);
        // TODO: populate detection result once UI flow is defined
      } catch (error) {
        console.error("Error detecting item from uploaded image:", error);
        toast({
          title: "Detection failed",
          description: "Please add the item manually.",
          variant: "destructive",
        });
      } finally {
        setIsDetectingItem(false);
      }
    };

    reader.readAsDataURL(file);
  };

  const pendingRequestedLabel = useMemo(() => {
    if (!pendingItem) return undefined;
    return formatQuantityLabel(
      pendingItem.requestedQuantity,
      pendingItem.requestedUnit ?? undefined
    );
  }, [pendingItem]);

  const pendingSuggestedLabel = useMemo(() => {
    if (!pendingItem) return undefined;
    if (pendingItem.suggestedAmountLabel) {
      return pendingItem.suggestedAmountLabel;
    }
    if (typeof pendingItem.suggestedQuantity === "number") {
      return formatQuantityLabel(
        pendingItem.suggestedQuantity,
        pendingItem.suggestedUnit ?? pendingItem.requestedUnit ?? undefined
      );
    }
    return undefined;
  }, [pendingItem]);

  return (
    <Layout>
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        <header className="mb-6">
          <h1 className="text-3xl font-bold text-foreground mb-2">Shopping List</h1>
          <p className="text-muted-foreground">Manage your shopping items</p>
        </header>

        {/* Action Buttons */}
        <div className="flex gap-2 mb-4">
          <Button
            onClick={handleMarkAllBought}
            className="flex-1"
            disabled={shoppingItems.length === 0 || isMarkingBought}
          >
            <ShoppingBag className="w-4 h-4 mr-2" />
            {isMarkingBought ? "Processing..." : "Mark All as Bought"}
          </Button>
        </div>

        {/* Add Item Form */}
        <Card className="p-4 mb-6">
          <div className="space-y-3">
            {/* Camera/Upload Options */}
            <div className="grid grid-cols-2 gap-2">
          <Button variant="outline" onClick={startCamera}>
                <Camera className="w-4 h-4 mr-2" />
                Camera
              </Button>
              <Button
                variant="outline"
                onClick={() => document.getElementById("upload-input-list")?.click()}
              >
                <Upload className="w-4 h-4 mr-2" />
                Upload
              </Button>
            </div>
            <input
              id="upload-input-list"
              type="file"
              accept="image/*"
              className="hidden"
            onChange={handleGalleryUpload}
            />

          {(cameraImage || galleryImage) && (
            <div className="rounded-lg border border-muted p-3 flex flex-col gap-2">
              <span className="text-sm font-medium text-muted-foreground">
                Image preview (remove or add item to clear)
              </span>
              <div className="relative w-full h-48 overflow-hidden rounded-md bg-muted">
                <img
                  src={cameraImage ?? galleryImage ?? undefined}
                  alt="Selected item preview"
                  className="w-full h-full object-cover"
                />
                {isDetectingItem && (
                  <div className="absolute inset-0 bg-background/70 flex items-center justify-center">
                    <Loader2 className="w-6 h-6 animate-spin text-primary" />
                  </div>
                )}
              </div>
              <div className="flex gap-2 justify-end">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setCameraImage(null);
                    setGalleryImage(null);
                  }}
                  disabled={isDetectingItem}
                >
                  Clear Image
                </Button>
              </div>
            </div>
          )}

            {/* Manual Add */}
            <div className="flex gap-2">
              <Input
                placeholder="Item name"
                value={newItemName}
                onChange={(e) => setNewItemName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleAdd()}
              />
              <Input
                type="number"
                step="0.01"
                min="0"
                placeholder="Amount"
                value={newItemAmount}
                onChange={(e) => setNewItemAmount(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleAdd()}
                className="w-28"
              />
              <Input
                placeholder="Unit (optional)"
                value={newItemUnit}
                onChange={(e) => setNewItemUnit(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleAdd()}
                className="w-32"
              />
              <Button onClick={handleAdd} size="icon" disabled={isSubmittingItem}>
                <Plus className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </Card>

        {/* Shopping Items Grid */}
        {isListLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-6 h-6 animate-spin text-primary" />
          </div>
        ) : shoppingItems.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            <ShoppingBag className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>Your shopping list is empty</p>
            <p className="text-sm">Add items using camera or manually</p>
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {shoppingItems.map((item) => (
              <Card key={item.id} className="p-4 relative group hover:shadow-lg transition-shadow">
                <button
                  onClick={() => handleDelete(item.id)}
                  className="absolute top-2 right-2 p-1.5 rounded-full bg-destructive/10 text-destructive opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <Trash2 className="w-4 h-4" />
                </button>

                <div className="flex flex-col items-center gap-2 text-center">
                  <div className="text-3xl mb-2">🛒</div>
                  <h3 className="font-semibold text-foreground">{item.item_name}</h3>
                  <p className="text-xs text-muted-foreground">Added on {item.date_added}</p>
                  <div className="mt-2 px-2 py-1 rounded-full bg-primary/10 text-primary text-sm font-medium">
                    {formatQuantityLabel(item.quantity_numeric, item.quantity_unit ?? undefined)}
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>

      {/* Waste Warning Dialog */}
      <AlertDialog open={Boolean(pendingItem)} onOpenChange={(open) => !open && !isSubmittingItem && setPendingItem(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Waste Warning</AlertDialogTitle>
            <AlertDialogDescription>
              {pendingItem?.wastedAmountLabel ? (
                <>
                  You previously wasted <strong>{pendingItem.wastedAmountLabel}</strong> of {" "}
                  <strong>{pendingItem.name}</strong>. Do you still want to add {" "}
                  <strong>{pendingRequestedLabel}</strong>?
                </>
              ) : (
                <>
                  This item has been wasted before. Do you still want to add {" "}
                  <strong>{pendingRequestedLabel}</strong>?
                </>
              )}
              {pendingSuggestedLabel && (
                <>
                  <br />
                  <br />
                  Suggested amount: <strong>{pendingSuggestedLabel}</strong>
                </>
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={handleUseSuggestedAmount} disabled={isSubmittingItem}>
              No (use suggested amount)
            </AlertDialogCancel>
            <AlertDialogAction onClick={handleConfirmAdd} disabled={isSubmittingItem}>
              Yes (keep original)
            </AlertDialogAction>
            <Button variant="ghost" onClick={handleCancelAdd} disabled={isSubmittingItem}>
              Close
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <Dialog open={isCameraModalOpen} onOpenChange={(open) => !open && handleCloseCameraModal()}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Capture Item Photo</DialogTitle>
            <DialogDescription>
              Align the item within the frame and tap capture when ready. If you don&apos;t see the camera feed,
              verify browser permissions or try switching devices.
            </DialogDescription>
          </DialogHeader>
          <div className="relative w-full h-64 overflow-hidden rounded-md bg-muted">
            {cameraError ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-center px-4">
                <p className="text-sm text-destructive font-medium">{cameraError}</p>
                <p className="text-xs text-muted-foreground mt-2">
                  Make sure your browser has permission to use the webcam. Some laptops require unplugging external
                  cameras or switching to a different browser.
                </p>
              </div>
            ) : (
              <video ref={videoRef} className="w-full h-full object-cover" autoPlay playsInline muted />
            )}
            {isDetectingItem && !cameraError && (
              <div className="absolute inset-0 bg-background/70 flex items-center justify-center">
                <Loader2 className="w-6 h-6 animate-spin text-primary" />
              </div>
            )}
          </div>
          <canvas ref={canvasRef} className="hidden" />
          <DialogFooter className="flex sm:justify-between gap-2">
            <Button variant="outline" onClick={handleCloseCameraModal} disabled={isDetectingItem}>
              Cancel
            </Button>
            <Button onClick={captureCameraFrame} disabled={isDetectingItem || Boolean(cameraError)}>
              Capture
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Layout>
  );
};

export default ShoppingList;
