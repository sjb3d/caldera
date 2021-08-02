extern crate proc_macro;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    braced, bracketed,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    token, Error, Ident, LitInt, Result, Token,
};

mod kw {
    syn::custom_keyword!(SampledImage);
    syn::custom_keyword!(StorageImage);
    syn::custom_keyword!(UniformData);
    syn::custom_keyword!(StorageBuffer);
    syn::custom_keyword!(AccelerationStructure);
}

enum BindingType {
    SampledImage,
    StorageImage,
    UniformData { ty: Ident },
    StorageBuffer,
    AccelerationStructure,
}

struct Binding {
    name: Ident,
    ty: BindingType,
    array: Option<usize>,
}

struct Layout {
    visibility: Option<Token![pub]>,
    name: Ident,
    bindings: Punctuated<Binding, token::Comma>,
}

impl Parse for BindingType {
    fn parse(input: ParseStream) -> Result<Self> {
        let lookahead = input.lookahead1();
        if lookahead.peek(kw::SampledImage) {
            input.parse::<kw::SampledImage>()?;
            Ok(BindingType::SampledImage)
        } else if lookahead.peek(kw::StorageImage) {
            input.parse::<kw::StorageImage>()?;
            Ok(BindingType::StorageImage)
        } else if lookahead.peek(kw::UniformData) {
            input.parse::<kw::UniformData>()?;
            input.parse::<token::Lt>()?;
            let ty = input.parse()?;
            input.parse::<token::Gt>()?;
            Ok(BindingType::UniformData { ty })
        } else if lookahead.peek(kw::StorageBuffer) {
            input.parse::<kw::StorageBuffer>()?;
            Ok(BindingType::StorageBuffer)
        } else if lookahead.peek(kw::AccelerationStructure) {
            input.parse::<kw::AccelerationStructure>()?;
            Ok(BindingType::AccelerationStructure)
        } else {
            Err(lookahead.error())
        }
    }
}

impl Parse for Binding {
    fn parse(input: ParseStream) -> Result<Self> {
        let name = input.parse()?;
        input.parse::<token::Colon>()?;
        let lookahead = input.lookahead1();
        let (ty, array) = if lookahead.peek(token::Bracket) {
            let content;
            let _bracket_token: token::Bracket = bracketed!(content in input);
            let ty = content.parse()?;
            content.parse::<token::Semi>()?;
            let array_lit = content.parse::<LitInt>()?;
            let array = array_lit.base10_parse()?;
            if !matches!(ty, BindingType::StorageImage) {
                return Err(Error::new(
                    content.span(),
                    "expected `StorageImage` for descriptor array",
                ));
            }
            if array == 0 {
                return Err(Error::new(array_lit.span(), "array length must be non-zero"));
            }
            (ty, Some(array))
        } else {
            let ty = input.parse()?;
            (ty, None)
        };
        Ok(Self { name, ty, array })
    }
}

impl Parse for Layout {
    fn parse(input: ParseStream) -> Result<Self> {
        let visibility = input.parse()?;
        let name = input.parse()?;
        let content;
        let _brace_token: token::Brace = braced!(content in input);
        let bindings = content.parse_terminated(Binding::parse)?;
        Ok(Self {
            visibility,
            name,
            bindings,
        })
    }
}

impl Binding {
    fn get_binding(&self) -> (Option<TokenStream2>, TokenStream2) {
        match self.ty {
            BindingType::SampledImage => {
                let sampler = format_ident!("{}_sampler", self.name);
                (
                    Some(quote!(#sampler: vk::Sampler)),
                    quote!(DescriptorSetLayoutBinding::SampledImage { sampler: #sampler }),
                )
            }
            BindingType::StorageImage => {
                let count = self.array.unwrap_or(1) as u32;
                (None, quote!(DescriptorSetLayoutBinding::StorageImage { count: #count }))
            }
            BindingType::UniformData { ref ty } => (
                None,
                quote!(DescriptorSetLayoutBinding::UniformData {
                    size: ::std::mem::size_of::<#ty>() as u32,
                }),
            ),
            BindingType::StorageBuffer => (None, quote!(DescriptorSetLayoutBinding::StorageBuffer)),
            BindingType::AccelerationStructure => (None, quote!(DescriptorSetLayoutBinding::AccelerationStructure)),
        }
    }

    fn get_data(&self) -> (TokenStream2, TokenStream2) {
        match self.ty {
            BindingType::SampledImage => {
                let image_view = format_ident!("{}_image_view", self.name);
                (
                    quote!(#image_view: vk::ImageView),
                    quote!(DescriptorSetBindingData::SampledImage { image_view: #image_view }),
                )
            }
            BindingType::StorageImage => {
                if let Some(count) = self.array {
                    let image_views = format_ident!("{}_image_views", self.name);
                    (
                        quote!(#image_views: &[vk::ImageView]),
                        quote!({
                            assert_eq!(#image_views.len(), #count);
                            DescriptorSetBindingData::StorageImage { image_views: #image_views }
                        }),
                    )
                } else {
                    let image_view = format_ident!("{}_image_view", self.name);
                    (
                        quote!(#image_view: vk::ImageView),
                        quote!(DescriptorSetBindingData::StorageImage { image_views: ::std::slice::from_ref(&#image_view) }),
                    )
                }
            }
            BindingType::UniformData { ref ty } => {
                let writer = format_ident!("{}_writer", self.name);
                (
                    quote!(#writer: impl Fn(&mut #ty)),
                    quote!(DescriptorSetBindingData::UniformData {
                        size: ::std::mem::size_of::<#ty>() as u32,
                        align: ::std::mem::align_of::<#ty>() as u32,
                        writer: &move |s: &mut [u8]| {
                            #writer(bytemuck::from_bytes_mut(s));
                        },
                    }),
                )
            }
            BindingType::StorageBuffer => {
                let buffer = format_ident!("{}_buffer", self.name);
                (
                    quote!(#buffer: vk::Buffer),
                    quote!(DescriptorSetBindingData::StorageBuffer { buffer: #buffer }),
                )
            }
            BindingType::AccelerationStructure => {
                let accel = format_ident!("{}_accel", self.name);
                (
                    quote!(#accel: vk::AccelerationStructureKHR),
                    quote!(DescriptorSetBindingData::AccelerationStructure { accel: #accel }),
                )
            }
        }
    }
}

#[proc_macro]
pub fn descriptor_set_layout(input: TokenStream) -> TokenStream {
    let Layout {
        visibility,
        name,
        bindings,
    } = parse_macro_input!(input as Layout);

    let (binding_args, binding_entries): (Vec<_>, Vec<_>) = bindings.iter().map(Binding::get_binding).unzip();
    let binding_args: Vec<_> = binding_args.iter().filter_map(|a| a.as_ref()).collect();
    let (data_args, data_entries): (Vec<_>, Vec<_>) = bindings.iter().map(Binding::get_data).unzip();

    quote!(
        #[repr(transparent)]
        #visibility struct #name(pub vk::DescriptorSetLayout);

        impl #name {
            pub fn new(descriptor_set_layout_cache: &mut DescriptorSetLayoutCache, #(#binding_args),*) -> Self {
                let bindings = &[#(#binding_entries),*];
                Self(descriptor_set_layout_cache.create_descriptor_set_layout(bindings))
            }

            #[allow(clippy::too_many_arguments)]
            pub fn write(&self, descriptor_pool: &DescriptorPool, #(#data_args),*) -> vk::DescriptorSet {
                let data = &[#(#data_entries),*];
                descriptor_pool.create_descriptor_set(self.0, data)
            }
        }
    )
    .into()
}
